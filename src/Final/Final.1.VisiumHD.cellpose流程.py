import bin2cell as b2c
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
import anndata
import os
import matplotlib.pyplot as plt
import pickle
from scipy.io import mmwrite
import skmisc
import celltypist
from celltypist import models
from matplotlib import rcParams
from matplotlib import font_manager
import seaborn as sns
import warnings
from matplotlib.patches import Patch
from cellpose import models, io
warnings.filterwarnings("ignore")
plt.set_loglevel('WARNING')



# ================= 1. 环境与路径设置 =================
sample_name = "Visium_HD_Human_Colon_Cancer_P2"
base_dir = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample_name}"
os.makedirs(f"{base_dir}/cellpose", exist_ok=True)

mpp = 0.5  # 缩放分辨率
he_tiff_path = f"{base_dir}/cellpose/he_05mpp.tiff"

# ================= 2. 数据预处理 =================
path = f"{base_dir}/binned_outputs/square_002um/"
source_image_path = f"{base_dir}/{sample_name}_tissue_image.btf"
spaceranger_image_path = f"{base_dir}/spatial"

adata = b2c.read_visium(path, 
                        source_image_path=source_image_path, 
                        spaceranger_image_path=spaceranger_image_path)
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_counts=1)
adata

# 生成缩放后的 HE 用于分割
b2c.scaled_he_image(adata, mpp=mpp, save_path=he_tiff_path)
b2c.destripe(adata)

# ================= 3. 使用 Cellpose 进行细胞核分割 =================
print("正在启动 Cellpose 运行...")
img = io.imread(he_tiff_path)

# 初始化模型：'nuclei' 效果最稳，如果想分细胞轮廓可用 'cyto2'
model = models.CellposeModel(gpu=True, model_type='cyto2')

# 运行分割
# diameter: 0.5mpp 下建议设为 20-30，None 则为自动估计
# 原代码：masks, flows, styles, diams = model.eval(...)
# 修改后：去掉了 diams
masks, flows, styles = model.eval(
    img, 
    diameter=25, 
    channels=[0, 0], 
    flow_threshold=0.4, 
    cellprob_threshold=0.0
)

# 将 Cellpose 的 Mask 转换为 bin2cell 兼容的 npz
cp_he_npz = f"{base_dir}/cellpose/he_cellpose.npz"
np.savez(cp_he_npz, labels=masks)

# ================= 4. 将 Cellpose 标签映射回 adata =================
# 映射 H&E 标签
b2c.insert_labels(adata, 
                  labels_npz_path=cp_he_npz, 
                  basis="spatial", 
                  spatial_key="spatial_cropped_150_buffer",
                  mpp=mpp, 
                  labels_key="labels_he")

# 扩张标签（覆盖细胞质）
b2c.expand_labels(adata, labels_key='labels_he', expanded_labels_key="labels_he_expanded")

# ================= 5. GEX 辅助分割 (可选替换为 Cellpose) =================
# 生成基因热力图
gex_tiff_path = f"{base_dir}/cellpose/gex_05mpp.tiff"
b2c.grid_image(adata, "n_counts_adjusted", mpp=mpp, sigma=5, save_path=gex_tiff_path)

# 对 GEX 图像也运行 Cellpose (使用荧光模型 'cyto')
img_gex = io.imread(gex_tiff_path)
model_gex = models.Cellpose(gpu=True, model_type='cyto')
masks_gex, _, _ = model_gex.eval(img_gex, diameter=30, channels=[0,0])

cp_gex_npz = f"{base_dir}/cellpose/gex_cellpose.npz"
np.savez(cp_gex_npz, labels=masks_gex)

b2c.insert_labels(adata, 
                  labels_npz_path=cp_gex_npz, 
                  basis="array", 
                  mpp=mpp, 
                  labels_key="labels_gex")

# ================= 6. 整合与生成单细胞对象 =================
# 整合 HE 和 GEX 的分割结果
b2c.salvage_secondary_labels(adata, 
                             primary_label="labels_he_expanded", 
                             secondary_label="labels_gex", 
                             labels_key="labels_joint")

# 转换为单细胞 AnnData
cdata = b2c.bin_to_cell(adata, labels_key="labels_joint", spatial_keys=["spatial", "spatial_cropped_150_buffer"])

# 保存结果
cdata.write(f"{base_dir}/cellpose/{sample_name}_cellpose_integrated.h5ad")


#%% celltypist 细胞类型预测
#cdata = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.h5ad")
cdata.var_names_make_unique()

cdata = cdata[cdata.obs['bin_count']>5] # min 5 bins

#need integers for seuratv3 hvgs
cdata.X.data = np.round(cdata.X.data)
cdata.raw = cdata.copy()

sc.pp.filter_genes(cdata, min_cells=3)
sc.pp.filter_cells(cdata,min_genes=100)
sc.pp.calculate_qc_metrics(cdata,inplace=True)

sc.pp.highly_variable_genes(cdata,n_top_genes=5000,flavor="seurat_v3")
sc.pp.normalize_total(cdata,target_sum=1e4)
sc.pp.log1p(cdata)


predictions_b2c_crc = celltypist.annotate(cdata, model = 'Human_Colorectal_Cancer.pkl', majority_voting = False)

cdata = predictions_b2c_crc.to_adata()

cdata_hvg = cdata[:, cdata.var["highly_variable"]]
sc.pp.scale(cdata_hvg, max_value=10)
sc.pp.pca(cdata_hvg, use_highly_variable=True)
sc.pp.neighbors(cdata_hvg)
sc.tl.umap(cdata_hvg)

# 整合对象
# 3. 核心整合：把 UMAP、PCA 和分类标签从 5000 基因的对象“搬”过去
cdata.obsm['X_umap'] = cdata_hvg.obsm['X_umap']
cdata.obsm['X_pca'] = cdata_hvg.obsm['X_pca']
cdata.obsm['spatial'] = cdata_hvg.obsm['spatial']
cdata.obsm['spatial_cropped_150_buffer'] = cdata_hvg.obsm['spatial_cropped_150_buffer']
cdata.obsp['distances'] = cdata_hvg.obsp['distances']
cdata.obsp['connectivities'] = cdata_hvg.obsp['connectivities']
cdata.uns.update(cdata_hvg.uns)

cdata.write(f"{base_dir}/cellpose/{sample_name}_cellpose_adata_annotated.h5ad")
