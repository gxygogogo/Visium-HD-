import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from scipy import sparse
from scipy.io import mmwrite


cell_adata = sc.read_h5ad('/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_cell_level_adata.h5ad')


# 1. 导出表达矩阵 (Matrix Market 格式，节省空间且保留稀疏性)
mmwrite("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/SeuratInput/counts.mtx", cell_adata.X.T) # 注意转置，Seurat 习惯 基因x细胞

# 2. 导出细胞注释 (obs)
cell_adata.obs.to_csv("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/SeuratInput/metadata.csv")

# 3. 导出基因注释 (var)
cell_adata.var.to_csv("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/SeuratInput/genes.csv")

# 4. 导出空间坐标 (spatial)
# 将坐标保存为 csv
spatial_coords = pd.DataFrame(
    cell_adata.obsm['spatial'], 
    index=cell_adata.obs_names, 
    columns=['x', 'y']
)
spatial_coords.to_csv("/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/SeuratInput/spatial_coords.csv")




import os
from scipy.io import mmwrite
import gzip

output_dir = "/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/filtered_feature_cell_matrix"
os.makedirs(output_dir, exist_ok=True)

# 1. 导出 matrix.mtx (并压缩)
# 注意：Seurat 需要 [Genes x Cells]
mmwrite(os.path.join(output_dir, "matrix.mtx"), cell_adata.X.T)

# 2. 导出 barcodes.tsv
cell_adata.obs_names.to_series().to_csv(os.path.join(output_dir, "barcodes.tsv"), 
                                        index=False, header=False, sep="\t")

# 3. 导出 features.tsv
# 包含 ID 和 Symbol 两列
features = cell_adata.var[['gene_ids']].copy()
features['symbol'] = cell_adata.var_names
features.to_csv(os.path.join(output_dir, "features.tsv"), 
                index=False, header=False, sep="\t")

# (可选) 使用 gzip 压缩这些文件，使其完全模仿官方格式
# !gzip my_segmented_matrix/*.mtx ...


