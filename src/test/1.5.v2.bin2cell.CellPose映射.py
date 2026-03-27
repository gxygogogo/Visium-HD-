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
with open(f"{data_dir}/output_cellpose/labels.pckl", "rb") as f:
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
cell_adata.write("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_cellpose_adata.h5ad")


#%% ================= 5. 导出为 Seurat 标准三件套 (方便 R 读取) =================
output_path = f"{data_dir}/filtered_feature_cell_matrix_cellpose"
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



