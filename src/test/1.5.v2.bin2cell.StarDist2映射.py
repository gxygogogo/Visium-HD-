import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
import anndata
import os
import pickle
from scipy.io import mmwrite

# ================= 1. 加载基础数据 =================
data_dir = "/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1"

# 加载原始 2um bin 数据
adata_bin = sc.read_10x_h5(f"{data_dir}/binned_outputs/square_002um/filtered_feature_bc_matrix.h5")

# 加载分割结果 labels.pckl (Mask 矩阵)
with open(f"{data_dir}/output_stardist/labels.pckl", "rb") as f:
    labels = pickle.load(f)

# 加载物理坐标
tissue_pos = pd.read_parquet(f"{data_dir}/binned_outputs/square_002um/spatial/tissue_positions.parquet").set_index('barcode')

# ================= 2. 空间坐标对齐与映射 (核心改进) =================
# 获取所有 bin 的像素坐标
# 注意：图像索引通常是 [row, col] -> [Y, X]
# 确保 pxl_row 和 pxl_col 没有因为图像缩放而产生比例偏差
all_bins_coords = tissue_pos.loc[adata_bin.obs_names, ['pxl_row_in_fullres', 'pxl_col_in_fullres']].values

# 直接从 labels 矩阵中提取每个 bin 对应的 Cell ID
# 使用 int32 避免越界或溢出
bin_cell_ids = labels[all_bins_coords[:, 0].astype(int), all_bins_coords[:, 1].astype(int)]

# 将 ID 存入 adata_bin 并过滤掉不在细胞内的 bins (ID=0)
adata_bin.obs['cell_id'] = bin_cell_ids
adata_sub = adata_bin[adata_bin.obs['cell_id'] > 0].copy()

print(f"有效 Bin 数量: {adata_sub.n_obs}, 涉及唯一细胞数: {adata_sub.obs['cell_id'].nunique()}")

# ================= 3. 高效矩阵聚合 =================
# 将 cell_id 转换为类别
cells_category = adata_sub.obs['cell_id'].astype('category')
cell_names = cells_category.cat.categories 

# 构建指示矩阵 (Indicator Matrix)
row_idx = cells_category.cat.codes.values
col_idx = np.arange(len(cells_category))
indicator_matrix = sparse.coo_matrix(
    (np.ones(len(row_idx)), (row_idx, col_idx)),
    shape=(len(cell_names), adata_sub.n_obs)
).tocsr()

# 聚合基因表达量
if not sparse.isspmatrix_csr(adata_sub.X):
    adata_sub.X = sparse.csr_matrix(adata_sub.X)
summed_counts = indicator_matrix @ adata_sub.X

# ================= 4. 创建 cell_adata 并计算坐标 =================
cell_adata = anndata.AnnData(
    X=summed_counts,
    obs=pd.DataFrame(index=cell_names.astype(str)),
    var=adata_sub.var
)

# 计算每个细胞的质心坐标
coords = tissue_pos.loc[adata_sub.obs_names, ['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
cell_coords = (indicator_matrix @ coords) / indicator_matrix.sum(axis=1).A
cell_adata.obsm['spatial'] = cell_coords

# 质量检查
cell_adata.obs['n_counts'] = cell_adata.X.sum(axis=1).A1
cell_adata.obs['n_genes'] = (cell_adata.X > 0).sum(axis=1).A1

# 过滤低质量细胞 (根据你的 40 基因现状，先设低一点，逐步调高)
# 建议至少保留 n_genes > 20 的细胞
cell_adata = cell_adata[cell_adata.obs['n_genes'] > 20].copy()
cell_adata.write("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_cell_level_adata.h5ad")






import squidpy as sq
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.colors as mcolors

# ================= 1. 计算聚类 (cell_adata 预处理) =================
# 如果你已经聚过类了，可以跳过这一步
sc.pp.normalize_total(cell_adata, target_sum=1e4)
sc.pp.log1p(cell_adata)
sc.pp.highly_variable_genes(cell_adata, n_top_genes=2000)
sc.pp.pca(cell_adata)
sc.pp.neighbors(cell_adata, n_neighbors=15, n_pcs=30)
sc.tl.leiden(cell_adata, resolution=0.5) # 生成 'leiden' 列


# ================= 2. 准备 ROI 数据与 Mask 裁剪 =================
# 定义你感兴趣的像素范围 (根据你的 tissue_positions 确定)
row_min, row_max = 15000, 20000 
col_min, col_max = 10000, 15000

# 加载并裁剪原始 labels.pckl
with open("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/output_stardist/labels.pckl", "rb") as f:
    full_labels = pickle.load(f)
roi_labels = full_labels[row_min:row_max, col_min:col_max]

# 获取 leiden 的类别和对应的颜色
clusters = cell_adata.obs['leiden'].cat.categories
# 获取 scanpy 自动生成的颜色列表 (例如 'leiden_colors')
if 'leiden_colors' in cell_adata.uns:
    cluster_colors = cell_adata.uns['leiden_colors']
else:
    # 如果没有，手动生成一组颜色
    import seaborn as sns
    cluster_colors = sns.color_palette("husl", len(clusters)).as_hex()

# 创建 ID 到 颜色的映射字典 (ID 必须是整数)
# 假设你的 cell_adata.obs.index 转成 int 后对应 labels 里的值
id_to_cluster = cell_adata.obs['leiden']
color_map_dict = {int(idx): col for idx, col in zip(cell_adata.obs.index, [cluster_colors[list(clusters).index(c)] for c in id_to_cluster])}

# ================= 2. 对 ROI 掩码进行重赋色 =================
# 加载 ROI 标签 (假设你已经裁剪好了 roi_labels)
# 如果没裁剪，请执行: roi_labels = full_labels[row_min:row_max, col_min:col_max]

# 创建一个和 ROI 同样大小的 RGB 图像矩阵 (H, W, 3)
# 初始化为全白色 [1, 1, 1] 或全黑色 [0, 0, 0]
h, w = roi_labels.shape
colored_roi = np.ones((h, w, 3)) 

# 遍历 ROI 中存在的唯一细胞 ID 并填色
unique_ids = np.unique(roi_labels)
for cell_id in unique_ids:
    if cell_id == 0: continue # 跳过背景
    if cell_id in color_map_dict:
        mask = (roi_labels == cell_id)
        # 将 HEX 颜色转为 RGB 分量
        rgb = mcolors.to_rgb(color_map_dict[cell_id])
        colored_roi[mask] = rgb

# ================= 3. 绘制并高清保存 =================
plt.figure(figsize=(15, 15))
plt.imshow(colored_roi, interpolation='nearest')
plt.axis('off')
plt.title(f"Cluster Mapping on Segmentation Mask", fontsize=20)

output_path = "/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/figures/Pure_Matplotlib_ROI.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()






#%% ================= 5. 导出为 Seurat 标准三件套 (方便 R 读取) =================
output_path = f"{data_dir}/filtered_feature_cell_matrix_2"
os.makedirs(output_path, exist_ok=True)

# 导出 matrix.mtx
mmwrite(f"{output_path}/matrix.mtx", cell_adata.X.T)

# 导出 barcodes.tsv
cell_adata.obs_names.to_series().to_csv(f"{output_path}/barcodes.tsv", index=False, header=False, sep="\t")

# 导出 features.tsv (Seurat 需要 Gene ID 和 Symbol)
features = pd.DataFrame({
    'id': cell_adata.var['gene_ids'] if 'gene_ids' in cell_adata.var else cell_adata.var_names,
    'symbol': cell_adata.var_names
})
features.to_csv(f"{output_path}/features.tsv", index=False, header=False, sep="\t")

# 导出空间坐标供 R 手动加载
pd.DataFrame(cell_adata.obsm['spatial'], index=cell_adata.obs_names, columns=['x', 'y']).to_csv(f"{output_path}/spatial_coords.csv")



