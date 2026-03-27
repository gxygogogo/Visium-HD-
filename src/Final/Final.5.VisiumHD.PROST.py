import pandas as pd
import numpy as np
import scanpy as sc
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import PROST
import h5py
PROST.__version__


# the location of R (used for the mclust clustering)
ENVpath = "/public1/xinyu/Software/Miniconda3/envs/PROST_ENV"            # refer to 'How to use PROST' section 
os.environ['R_HOME'] = f'{ENVpath}/lib/R'
os.environ['R_USER'] = f'{ENVpath}/lib/python3.7/site-packages/rpy2'

# init
SEED = 818
PROST.setup_seed(SEED)

sample_name = "Visium_HD_Human_Colon_Cancer_P5"
base_dir = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample_name}"
#os.makedirs(f"{base_dir}/stardist", exist_ok=True)

path = f"{base_dir}/binned_outputs/square_002um/"
#the image you used for --image of spaceranger, that's the one the spatial coordinates are based on
source_image_path = f"{base_dir}/{sample_name}_tissue_image.btf"
spaceranger_image_path = f"{base_dir}/spatial"

cdata = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.v2.h5ad")

# 在新版环境执行
for col in cdata.obs.columns:
    if cdata.obs[col].dtype.name == 'category':
        cdata.obs[col] = cdata.obs[col].astype(str)

# 强制不使用新的 H5AD 规范保存
cdata.write_h5ad(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.fixed_for_prost.h5ad")

cdata = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.fixed_for_prost.h5ad")


cdata = PROST.prepare_for_PI(cdata, percentage = 0.01, platform="Visium HD")
cdata = PROST.cal_PI(cdata, kernel_size=6, platform="Visium HD")

# Calculate spatial autocorrelation statistics and Hypothesis test
'''
PROST.spatial_autocorrelation(adata, k = 10, permutations = None)
'''

cdata.write_h5ad(f"{base_dir}/stardist/PROST/PI_result.h5")


