#%% 导入必要的库
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import tifffile
import geopandas as gpd
from shapely.geometry import Point
from cellpose import models, core, plot, io
import anndata
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__() # 将限制扩大


#%% 函数
# 定义读取图像的函数，并可选择下采样因子
def read_dapi_image(img_path: str, downscale_factor: int = 2) -> np.ndarray:
    # 替换 io.imread 为 tifffile.imread
    img = tifffile.imread(img_path) 
    
    if img is None:
        raise ValueError(f"无法读取图像，请检查路径: {img_path}")
    
    print(f"原始图像形状: {img.shape}")
    
    # 如果图像是 (C, H, W) 格式，cellpose 通常需要 (H, W, C)
    # 检查并调整轴（Visium HD 图像有时是 3 维的）
    if img.ndim == 3 and img.shape[0] < 10:  # 假设通道在第一位
        img = np.transpose(img, (1, 2, 0))
        
    return downscale_image(img, downscale_factor=downscale_factor)
 
# 定义下采样图像的函数
def downscale_image(img: np.ndarray, downscale_factor: int = 2) -> np.ndarray:
    # 计算每个轴上所需的填充量
    pad_height = (downscale_factor - img.shape[0] % downscale_factor) % downscale_factor
    pad_width = (downscale_factor - img.shape[1] % downscale_factor) % downscale_factor
    new_shape = (img.shape[0] + pad_height, img.shape[1] + pad_width, img.shape[2])  # 新图像形状
    new_image = np.zeros(new_shape, dtype=img.dtype)  # 创建新的零填充图像
    for channel in range(img.shape[2]):
        new_image[:, :, channel] = np.pad(
            img[:, :, channel], ((0, pad_height), (0, pad_width)), mode="constant"
        )  # 对每个通道进行零填充
    return new_image  # 返回填充后的图像
 
# 定义运行Cellpose进行细胞分割的函数
def run_cellpose(img: np.ndarray, model_path: str):
    use_GPU = core.use_gpu()  # 检查是否使用GPU
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)  # 加载Cellpose模型
    channels = [0, 0]  # 设置通道（这里假设为单通道）
    masks, flows, styles = model.eval(
        [img],
        channels=channels,
        diameter=model.diam_labels,
        flow_threshold=1,
        cellprob_threshold=0,
        batch_size=4,
    )  # 执行细胞分割并返回掩膜、流和样式信息
    return (masks, flows, styles)  # 返回分割结果


#%% CellPose 分割
sample = 'Visium_HD_Human_Colon_Cancer_P2'

# 设置Visium HD数据的文件路径并读取图像
img_path = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_tissue_image.btf"
maxed_visium = read_dapi_image(img_path, downscale_factor=1)  # 读取并下采样图像

# 设置分块参数和模型路径
chunk_per_axis = 2  # 每个轴上分块数目
mp = 'cyto'  # 使用细胞质模型进行分割

# 对图像进行分块并调用Cellpose进行细胞分割
masks_top_left, flows, styles = run_cellpose(
    maxed_visium[
        : np.shape(maxed_visium)[0] // chunk_per_axis,
        : np.shape(maxed_visium)[1] // chunk_per_axis,
        :,
    ],
    model_path=mp,
)
np.save(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/masks_top_left.cyto.masks.npy', masks_top_left)
del masks_top_left, flows, styles

masks_top_right, flows, styles = run_cellpose(
    maxed_visium[
        : np.shape(maxed_visium)[0] // chunk_per_axis,
        np.shape(maxed_visium)[1] // chunk_per_axis :,
        :,
    ],
    model_path=mp,
)
np.save(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/masks_top_right.cyto.masks.npy', masks_top_right)
del masks_top_right, flows, styles

masks_bottom_left, flows, styles = run_cellpose(
    maxed_visium[
        np.shape(maxed_visium)[0] // chunk_per_axis :,
        : np.shape(maxed_visium)[1] // chunk_per_axis,
        :,
    ],
    model_path=mp,
)
np.save(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/masks_bottom_left.cyto.masks.npy', masks_bottom_left)
del masks_bottom_left, flows, styles

masks_bottom_right, flows, styles = run_cellpose(
    maxed_visium[
        np.shape(maxed_visium)[0] // chunk_per_axis :,
        np.shape(maxed_visium)[1] // chunk_per_axis :,
        :,
    ],
    model_path=mp,
)
np.save(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/masks_bottom_right.cyto.masks.npy', masks_bottom_right)
del masks_bottom_right, flows, styles

masks_top_left = np.load(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/masks_top_left.cyto.masks.npy')
masks_top_right = np.load(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/masks_top_right.cyto.masks.npy')
masks_bottom_left = np.load(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/masks_bottom_left.cyto.masks.npy')
masks_bottom_right = np.load(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/masks_bottom_right.cyto.masks.npy')

# 初始化全掩膜数组，准备合并分块结果
constant = 1000000  # 用于区分不同块的常量值
full_mask = np.zeros_like(maxed_visium[:, :, 0], dtype=np.uint32)  # 创建与原图相同大小的掩膜数组
 
# 将各个块的掩膜合并到全掩膜中，并加上常量以区分不同区域
full_mask[
    : np.shape(maxed_visium)[0] // chunk_per_axis,
    : np.shape(maxed_visium)[1] // chunk_per_axis,
] = masks_top_left[0]
full_mask[
    : np.shape(maxed_visium)[0] // chunk_per_axis,
    np.shape(maxed_visium)[1] // chunk_per_axis :,
] = masks_top_right[0] + (constant)
full_mask[
    np.shape(maxed_visium)[0] // chunk_per_axis :,
    : np.shape(maxed_visium)[1] // chunk_per_axis,
] = masks_bottom_left[0] + (constant * 2)
full_mask[
    np.shape(maxed_visium)[0] // chunk_per_axis :,
    np.shape(maxed_visium)[1] // chunk_per_axis :,
] = masks_bottom_right[0] + (constant * 3)
 
# 将全掩膜中值为常量的部分设置为0，表示没有细胞
full_mask = np.where(full_mask % constant == 0, 0, full_mask)

# 将全掩膜保存为PNG格式文件
tifffile.imsave(f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/visium_hd_segmentation.png", full_mask)



#%% 将Cellpose分割结果与空间转录组数据进行整合
# 设置Visium HD数据目录路径以加载空间转录组数据
dir_base = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/binned_outputs/square_002um"

# 加载Visium HD数据中的基因表达矩阵（HDF5格式）
raw_h5_file = f"{dir_base}/filtered_feature_bc_matrix.h5"
adata = sc.read_10x_h5(raw_h5_file)  # 使用Scanpy读取数据

# 加载组织位置坐标文件（Parquet格式）
tissue_position_file = f"{dir_base}/spatial/tissue_positions.parquet"
df_tissue_positions = pd.read_parquet(tissue_position_file)

# 将数据框的索引设置为条形码，以便后续合并操作使用
df_tissue_positions = df_tissue_positions.set_index("barcode")

# 在数据框中创建索引以检查合并情况
df_tissue_positions["index"] = df_tissue_positions.index

# 将组织位置添加到AnnData对象的元数据中（obs）
adata.obs = pd.merge(adata.obs, df_tissue_positions, left_index=True, right_index=True)

# 从坐标数据框创建GeoDataFrame，以便进行空间分析
geometry = [
    Point(xy)
    for xy in zip(
        df_tissue_positions["pxl_col_in_fullres"],
        df_tissue_positions["pxl_row_in_fullres"],
    )
]
gdf_coordinates = gpd.GeoDataFrame(df_tissue_positions, geometry=geometry)

## 将捕获区域分配给单个细胞
# 检查点掩膜是否在全掩膜范围内，以确保坐标有效性
point_mask = (
    gdf_coordinates["pxl_row_in_fullres"].values.astype(int) < np.shape(full_mask)[0]
) & (gdf_coordinates["pxl_col_in_fullres"].values.astype(int) < np.shape(full_mask)[1])

# 根据有效坐标从全掩膜中提取细胞ID信息
cells = full_mask[
    gdf_coordinates["pxl_row_in_fullres"].values.astype(int)[point_mask],
    gdf_coordinates["pxl_col_in_fullres"].values.astype(int)[point_mask],
]
gdf_coordinates = gdf_coordinates[point_mask]
gdf_coordinates["cells"] = cells  # 将细胞ID添加到GeoDataFrame中
assigned_regions = gdf_coordinates[gdf_coordinates["cells"] > 0]

# 将细胞ID信息合并到AnnData对象中，以便后续分析使用
adata.obs = adata.obs.merge(
    assigned_regions[["cells"]], left_index=True, right_index=True, how="left"
)
adata = adata[~pd.isna(adata.obs["cells"])]  # 移除没有细胞ID的数据行
adata.obs["cells"] = adata.obs["cells"].astype(int)  # 将细胞ID转换为整数类型

## --- 在捕获区域内汇总转录本计数 ---
groupby_object = adata.obs.groupby(["cells"], observed=True)
counts = adata.X  
spatial_coords = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values # 注意顺序：X, Y
N_groups = groupby_object.ngroups
N_genes = counts.shape[1]

summed_counts = sparse.lil_matrix((N_groups, N_genes))
polygon_id = []
cell_coords = [] # 用于存储每个细胞的质心坐标
row = 0

for polygons, idx_ in groupby_object.indices.items():
    summed_counts[row] = counts[idx_].sum(0)
    # 计算该细胞包含的所有 bin 的坐标平均值，作为细胞的中心点
    cell_coords.append(np.mean(spatial_coords[idx_], axis=0))
    row += 1
    polygon_id.append(polygons)

cell_coords = np.array(cell_coords)
summed_counts = summed_counts.tocsr()

polygon_id_np = np.array(polygon_id)

grouped_adata = anndata.AnnData(
    X=summed_counts,
    obs=pd.DataFrame(index=polygon_id_np.astype(str)), # 现在可以正常转换了
    var=adata.var,
)

# 解决 Variable names are not unique 的警告
grouped_adata.var_names_make_unique()

grouped_adata.obs['cells'] = polygon_id_np
grouped_adata.obsm['spatial'] = cell_coords

# 保存
grouped_adata.write(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_adata_cellpose.h5ad')


#%% 降分辨率
## 降分辨率
import cv2

# 将 full_mask 缩放到原来的 1/10 宽度和高度
preview_scale = 0.3
width = int(full_mask.shape[1] * preview_scale)
height = int(full_mask.shape[0] * preview_scale)
# 使用 INTER_NEAREST 保证细胞 ID 不会被插值搞乱
preview_mask = cv2.resize(full_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)

plt.figure(figsize=(15, 15))
plt.imshow(preview_mask > 0, cmap='gray') # 只看有没有分割出东西
plt.title("Segmentation Preview")
plt.savefig(f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_segmentation_check.png")

tifffile.imwrite(f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_segmentation_check.tiff", preview_mask)





#%%
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
adata = sc.read_h5ad(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_adata_cellpose.h5ad')

# 1. 基础过滤（可选，根据你的细胞质量决定）
sc.pp.filter_cells(adata, min_genes=10)

# 2. 预处理
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 3. 识别高变基因（必须，为了后续降维）
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# 4. PCA 降维（Leiden 需要基于 PCA 空间构建图）
sc.tl.pca(adata)

# 5. 计算邻居图 (这一步会生成 adata.uns['neighbors'])
# 20万细胞建议 n_pcs 用 30-50
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)

# 6. 现在可以运行聚类了
sc.tl.leiden(adata, resolution=0.5)

# 7. 计算 UMAP 用于可视化
sc.tl.umap(adata)

# 绘制并保存 UMAP
sc.pl.umap(adata, color='leiden', show=False) # 必须设 show=False
plt.savefig(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_leiden_clustering.png', dpi=300, bbox_inches='tight')
plt.close() # 习惯性关闭画布防止内存溢出

# 绘制并保存 空间图
sc.pl.spatial(adata, color=['leiden', 'EPCAM'], spot_size=1.5, show=False)
plt.savefig(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_spatial_expression.pdf', bbox_inches='tight')
plt.close()



#%% 原位展示基因关系
import scanpy as sc
import matplotlib.pyplot as plt

# 确保数据已经过标准化和对数化
# 如果还没做，请执行：
# sc.pp.normalize_total(grouped_adata, target_sum=1e4)
# sc.pp.log1p(grouped_adata)

# 绘制全局空间表达图
# spot_size 在 HD 数据中建议设置得非常小
sc.pl.spatial(
    grouped_adata, 
    color=['EPCAM'], # 展示基因种类数和特定标记基因
    spot_size=20,                      # 这里的 size 需要根据你的像素坐标系调整，通常 5-20
    frameon=False,
    cmap='magma',
    title=['EPCAM (Epithelial)'],
    show=False
)
plt.savefig(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/Global_InSitu_Expression.png', dpi=800)



#%% 统计
import numpy as np
import pandas as pd

# 统计每个细胞检测到的基因数量 (n_genes)
# adata.X != 0 会返回一个布尔矩阵，.sum(axis=1) 统计每行 True 的个数
n_genes = (grouped_adata.X > 0).sum(axis=1)

# 如果你的 X 是稀疏矩阵（常见情况），转换为 A 再计算
if isinstance(n_genes, np.matrix):
    n_genes = n_genes.A1 # 转换为 1D 数组

# 将结果存入 obs，方便后续绘图和过滤
grouped_adata.obs['n_genes'] = n_genes

# 打印统计摘要，查看每个细胞检测到的基因种类数
print(grouped_adata.obs['n_genes'].describe())

