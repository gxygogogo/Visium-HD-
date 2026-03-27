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
mapping_df = pd.read_csv("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/output_cellpose/Cellpose_Nuclei_Barcode_Map.csv")

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
