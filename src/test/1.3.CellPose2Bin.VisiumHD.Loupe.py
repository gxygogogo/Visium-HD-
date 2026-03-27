import pandas as pd

# 1. 准备要导出的数据框
# 至少包含 Barcode (这里是 cells ID) 和你要查看的分组信息
export_df = pd.DataFrame(index=grouped_adata.obs.index)

# 导出 cells ID (用于跟 Mask 对应，如果需要的话)
export_df['Cell_ID'] = grouped_adata.obs['cells'].values

# 导出 Leiden 聚类结果
if 'leiden' in grouped_adata.obs.columns:
    export_df['Leiden_Cluster'] = grouped_adata.obs['leiden'].values

# 导出基因表达量 (例如 EPCAM)
# 注意：Loupe 主要用于查看分类数据，基因表达量通常直接在 Loupe 里搜索
# 但你可以将表达量作为一个连续型的“特征”导入
gene_name = 'EPCAM'
if gene_name in grouped_adata.var_names:
    # 获取标准化后的表达量
    exp_data = grouped_adata[:, gene_name].X.toarray().flatten()
    export_df[f'{gene_name}_Expression'] = exp_data

# 2. 将索引重命名为 'Barcode'，这是 Loupe 识别的要求
export_df.index.name = 'Barcode'

# 3. 导出为 CSV
export_path = f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_loupe_import.csv'
export_df.to_csv(export_path)

