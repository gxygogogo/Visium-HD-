import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
import anndata
import os
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as mcolors
from skimage.segmentation import find_boundaries

#%% 2μm bin 数据映射到细胞水平的分析
# 加载原始的 2um bin 数据
# 这是从 SpaceRanger 输出的包含几百万个 bins 的对象
adata_bin = sc.read_10x_h5("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/binned_outputs/square_002um/filtered_feature_bc_matrix.h5")

# 加载你生成的映射表
mapping_df = pd.read_csv("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/output_stardist/Nuclei_Barcode_Map.csv")

# 将映射表设为以 barcode 为索引
mapping_df = mapping_df.set_index('index') # 注意这里的 'index' 是 csv 里的列名，代表 barcode

# 1. 检查并去除 mapping_df 中的重复索引（Barcode）
# keep='first' 表示如果一个 barcode 属于多个细胞，只保留第一个
mapping_df_unique = mapping_df.loc[~mapping_df.index.duplicated(keep='first')]

# 2. 重新确定交集（确保万无一失）
common_barcodes = adata_bin.obs_names.intersection(mapping_df_unique.index)

# 3. 重新切片 adata
adata_sub = adata_bin[common_barcodes].copy()

# 4. 执行赋值 (现在 index 是唯一的，不会再报错)
adata_sub.obs['cell_id'] = mapping_df_unique.loc[common_barcodes, 'id']

print(f"成功分配 ID。原始 bin 数: {len(common_barcodes)}, 涉及的唯一细胞数: {adata_sub.obs['cell_id'].nunique()}")

# 找出既在表达矩阵中又在映射表中的 Barcode
common_barcodes = adata_bin.obs_names.intersection(mapping_df_unique.index)
adata_sub = adata_bin[common_barcodes].copy()

# 将细胞 ID (id) 分配给 adata_sub 的 obs
adata_sub.obs['cell_id'] = mapping_df_unique.loc[common_barcodes, 'id']

# 1. 将 cell_id 转换为类别类型
cells_category = adata_sub.obs['cell_id'].astype('category')
cell_names = cells_category.cat.categories # 获取所有唯一的细胞 ID

# 2. 构建一个“指示矩阵” (Indicator Matrix)
# 这个矩阵的大小是 [唯一细胞数 x 原始 bins 数]
# 如果第 j 个 bin 属于第 i 个细胞，则矩阵元素 (i, j) 为 1
row_idx = cells_category.cat.codes.values
col_idx = np.arange(len(cells_category))
indicator_matrix = sparse.coo_matrix(
    (np.ones(len(row_idx)), (row_idx, col_idx)),
    shape=(len(cell_names), adata_sub.n_obs)
).tocsr()

# 3. 通过矩阵乘法求和，计算每个细胞的总基因表达量
# [细胞 x bins] * [bins x 基因] = [细胞 x 基因]
summed_counts = indicator_matrix @ adata_sub.X

# 4. 创建新的 AnnData 对象
cell_adata = anndata.AnnData(
    X=summed_counts,
    obs=pd.DataFrame(index=cell_names),
    var=adata_sub.var
)

# 5. 计算细胞的中心位置 (Spatial Coordinates)
# 我们取属于该细胞的所有 bins 坐标的平均值
tissue_pos = pd.read_parquet("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/binned_outputs/square_002um/spatial/tissue_positions.parquet").set_index('barcode')
coords = tissue_pos.loc[common_barcodes, ['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
cell_coords = (indicator_matrix @ coords) / indicator_matrix.sum(axis=1).A # 计算加权平均坐标

cell_adata.obsm['spatial'] = cell_coords

# 简单的质量核查
cell_adata.obs['n_counts'] = cell_adata.X.sum(axis=1).A1
cell_adata.obs['n_genes'] = (cell_adata.X > 0).sum(axis=1).A1

print(f"创建成功！新对象包含 {cell_adata.n_obs} 个细胞和 {cell_adata.n_vars} 个基因。")
print(cell_adata.obs[['n_counts', 'n_genes']].describe())

# 保存为 h5ad 文件供后续分析使用
cell_adata.write("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_cell_level_adata.h5ad")
cell_adata.write_loom("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_cell_level_adata.loom")



#%% 基于整合之后的 cell_adata 进行空间映射
# ================= 预处理 (必须) =================
# 在绘图前，必须对原始计数进行标准化和对数化，
# 否则不同细胞间的测序深度差异会掩盖真实的表达信号。
print("正在进行预处理 (Normalize & Log)...")
sc.pp.normalize_total(cell_adata, target_sum=1e4)
sc.pp.log1p(cell_adata)

# ================= 空间表达绘图 =================
# 我们在物理坐标（obsm['spatial']）上绘制基因表达
genes_to_plot = 'PIEZO1'

print(f"正在绘制 {genes_to_plot} 的空间表达图...")

# 尝试不同的 spot_size。
# 对于 10万+ 细胞的全图，通常在 5 到 30 之间。
# 值越小，越能体现单细胞分辨率；值越大，信号越连续。
spot_size_value = 15

# 使用 magma 色阶，黑色表示无表达，亮黄色表示高表达，对比度高。
sc.pl.spatial(
    cell_adata, 
    color=genes_to_plot, 
    spot_size=spot_size_value, 
    frameon=False,  # 关闭坐标轴边框
    cmap='magma',   # 设置色阶
    title=[f'{gene} Expression' for gene in genes_to_plot],
    show=False      # 不立即在终端显示，以便保存
)

# ================= 保存图片 =================
output_fig = f"/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_PIEZO1_Spatial.png"
plt.savefig(output_fig, dpi=600, bbox_inches='tight')
plt.close() # 关闭画布节省内存


## 绘制ROI区域
x_min, x_max = 12000, 18000
y_min, y_max = 12000, 18000

# 2. 从 cell_adata 中筛选位于该范围内的细胞
spatial_coords = cell_adata.obsm['spatial']
roi_mask = (spatial_coords[:, 0] >= x_min) & (spatial_coords[:, 0] <= x_max) & \
           (spatial_coords[:, 1] >= y_min) & (spatial_coords[:, 1] <= y_max)

cell_adata_roi = cell_adata[roi_mask].copy()

print(f"ROI 内的细胞数量: {cell_adata_roi.n_obs}")

# 3. 绘制 ROI 表达图
# 在局部图中，spot_size 可以适当调大（比如 30-60），让细胞看起来更饱满
sc.pl.spatial(
    cell_adata_roi, 
    color='PIEZO1', 
    spot_size=40, 
    cmap='magma',
    frameon=True,
    title='PIEZO1'
)
output_fig = f"/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_PIEZO1_roi_Spatial.png"
plt.savefig(output_fig, dpi=600, bbox_inches='tight')
plt.close() # 关闭画布节省内存


#%% 像素级映射
# 2. 加载 StarDist 生成的 labels
with open('/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/output_stardist/labels.pckl', 'rb') as f:
    labels = pickle.load(f)

# 3. 确定要绘制的基因
gene_name = 'PIEZO1' 

# 创建LookUP table
# 提取该基因在所有细胞中的表达值
# 注意：adata.obs 里的 ID 顺序需要与 labels 里的数值对应
# 假设 adata.obs.index 里的 'ID_1' 对应 labels 里的数值 1
exp_values = cell_adata[:, gene_name].X.toarray().flatten()

# 创建一个查找表 (LUT)
# 索引 0 留给背景，从 1 开始对应细胞 ID
lut = np.zeros(labels.max() + 1, dtype=np.float32)

# 将表达值填充到 LUT 中
# 注意：这里需要根据你 index 的命名规则解析出数字索引
# 示例：如果 index 是 'ID_1', 'ID_2'...
indices = [int(i.split('_')[1]) for i in cell_adata.obs_names]
lut[indices] = exp_values

# 映射与边界计算
# 将 labels 替换为表达量图
gene_img = lut[labels]

# 计算细胞边界 (用于叠加轮廓)
# mode='inner' 会在细胞内部边缘画线，不会模糊边界
boundaries = find_boundaries(labels, mode='inner')

# 绘图
# 定义显示范围 (ROI)
r0, r1, c0, c1 = 12000, 18000, 12000, 18000 # 根据你的组织位置调整

plt.figure(figsize=(12, 10), dpi=300)

# 1. 绘制表达背景
v_max = np.percentile(exp_values, 99) # 增强对比度
plt.imshow(gene_img[r0:r1, c0:c1], cmap='magma', vmin=0, vmax=v_max, interpolation='nearest')

# 2. 叠加白色轮廓
# 创建一个透明到白色的自定义 colormap
contour_cmap = mcolors.ListedColormap(['none', 'white'])
plt.imshow(boundaries[r0:r1, c0:c1], cmap=contour_cmap, alpha=0.5)

plt.title(f'Cellular Expression Contour: {gene_name}')
plt.colorbar(label='Log Normalized Expression')
plt.axis('off')
plt.savefig('/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Cell_Contour_Expression.png', bbox_inches='tight', dpi=600)
plt.show()

#%% 最终版像素级细胞基因表达映射
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.segmentation import find_boundaries

gene_name = 'PIEZO1' 

# 1. 提取表达数据
exp_vector = cell_adata[:, gene_name].X.toarray().flatten()

# 2. 建立 Lookup Table (包含背景 0)
lut = np.zeros(labels.max() + 1, dtype=np.float32)
# 确保 ID 映射逻辑一致
cell_indices = [int(str(name).split('_')[-1]) for name in cell_adata.obs_names]
lut[cell_indices] = exp_vector

# 3. 生成像素级图像
gene_img = lut[labels].astype(float)

# --- 关键步骤：将非细胞区域（背景）设为 NaN，以便显示白色 ---
gene_img[labels == 0] = np.nan 

# 4. 计算所有细胞轮廓
boundaries = find_boundaries(labels, mode='inner')

# 5. 绘图设置
r0, r1, c0, c1 = 12000, 18000, 12000, 18000 
plt.figure(figsize=(12, 12), facecolor='white') # 设置画布背景为白色

# 设置颜色上限 (vmax)
v_max = np.percentile(exp_vector, 99.5) if np.max(exp_vector) > 0 else 1.0

# --- 第一层：基因表达填充层 ---
# cmap 建议换成 'viridis' 或 'Reds'，在白色背景下视觉效果更好
# 如果坚持用 'magma'，0 值会偏黑紫色
current_cmap = plt.cm.magma.copy()
current_cmap.set_bad(color='white') # 将 NaN 区域（背景）显式设为白色

im = plt.imshow(gene_img[r0:r1, c0:c1], 
                cmap=current_cmap, 
                vmin=0, 
                vmax=8, 
                interpolation='nearest')

# --- 第二层：所有细胞的轮廓层 ---
# 在白色背景下，使用深灰色轮廓 (#333333) 效果最好
contour_cmap = mcolors.ListedColormap(['none', '#333333']) 
plt.imshow(boundaries[r0:r1, c0:c1], cmap=contour_cmap, alpha=0.4)

# 6. 图表装饰
plt.title(f'Cellular Expression: {gene_name}\n(White Background, Unified Scale)', 
          color='black', fontsize=15, pad=20)

cb = plt.colorbar(im, fraction=0.046, pad=0.04)
cb.set_label('Log Normalized Expression', color='black')
cb.ax.yaxis.set_tick_params(color='black', labelcolor='black')

plt.axis('off')

# 保存结果
output_path = f'/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_{gene_name}_WhiteBG_Infill.png'
plt.savefig(output_path, bbox_inches='tight', dpi=600, facecolor='white')
plt.show()


