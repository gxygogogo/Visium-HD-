import sys
import os

# 1. 关键：添加 CellDAG 的“父目录”
# 这样 Python 就能把 CellDAG 识别为一个 package
parent_path = "/public3/Xinyu/3D_tissue"
if parent_path not in sys.path:
    sys.path.append(parent_path)

# 2. 必须从包名开始导入
# 这样 model.py 内部的 "from .train_func" 才能被正确识别
from CellDAG import model as dt

# 3. 其他常用库
import torch
import scanpy as sc
import numpy as np

print("CellDAG 模型加载成功！")
import matplotlib.pyplot as plt




sample_name = "Visium_HD_Human_Colon_Cancer_P5"
base_dir = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample_name}"
#os.makedirs(f"{base_dir}/stardist", exist_ok=True)

path = f"{base_dir}/binned_outputs/square_002um/"
#the image you used for --image of spaceranger, that's the one the spatial coordinates are based on
source_image_path = f"{base_dir}/{sample_name}_tissue_image.btf"
spaceranger_image_path = f"{base_dir}/spatial"

save_path = f"{base_dir}/stardist/CellDAG"
if not os.path.exists(save_path): os.makedirs(save_path)



cdata = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_scNiche_adata.h5ad")

# ==================== 1. 纯 CD31 表达量筛选 ====================
# 确定基因名称
gene_name = 'PECAM1'

# 提取 CD31 表达量并进行筛选
# 建议先对原始计数进行标准化，以确保筛选阈值的生物学意义
# 如果 cdata 已经是经过 log1p 的，直接根据数值筛选即可
cd31_expr = cdata[:, gene_name].X
if hasattr(cd31_expr, "toarray"):
    cd31_expr = cd31_expr.toarray().flatten()

# 设定阈值：筛选表达量在前 95% 分位数以上或者大于某个值的细胞
# 这里我们采用保守的 > 0 筛选，或者你可以设为 > 0.5 来提高纯度
mask = cd31_expr > np.quantile(cd31_expr,0.99)
st_data_use = cdata[mask, :].copy()
print(f"筛选完成：基于 {gene_name} 表达量保留了 {st_data_use.n_obs} 个细胞")

# 1. 常规预处理后的 UMAP 观察
sc.pp.neighbors(st_data_use, n_neighbors=15, use_rep='X_pca')
sc.tl.umap(st_data_use)

res = 0.5 
sc.tl.leiden(st_data_use, resolution=res, key_added=f'leiden_res_{res}')

# 2. 对比查看：空间分布 vs UMAP 分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.embedding(st_data_use, 
                basis="spatial_cropped_150_buffer", 
                color=gene_name, 
                ax=ax1, 
                show=False)
sc.pl.umap(st_data_use, 
           color=gene_name, 
           ax=ax2, 
           show=False)
plt.savefig(f'{save_path}/{sample_name}_CD31_umap.pdf', dpi=1000, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

sc.pl.umap(
    st_data_use, 
    color=f'leiden_res_{res}', 
    title=f"UMAP Clusters (Res {res})",
    ax=ax,  # 明确指定画在当前的 ax 上
    show=False
)

plt.savefig(f'{save_path}/{sample_name}_leiden_umap.pdf', dpi=1000, bbox_inches='tight')
plt.close()


# ==================== 2. 模型训练准备 ====================
# 针对筛选后的细胞进行高变基因筛选，以提高轨迹推断的灵敏度
sc.pp.highly_variable_genes(st_data_use, n_top_genes=2000)
st_data_use = st_data_use[:, st_data_use.var['highly_variable']].copy()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = {
    "num_input": st_data_use.n_vars,
    "num_emb": 256,
    "dk_re": 16,
    "nheads": 1,
    "droprate": 0.15,
    "device": device,
    "neighbor_type": "noextern",
    "n_neighbors": 9,
    "num_epoch1": 1000,
    "num_epoch2": 1000,
    "lr": 0.001,
    "SEED": 24,
    "alpha": 1.0,
    "beta": 0.1,
    "theta1": 0.1,
    "theta2": 0.1,
    
    # --- 之前缺失并导致报错的参数 ---
    "leakyalpha": 0.1,       # 对应源码 351 行
    "resalpha": 0.5,         # 对应源码 353 行
    "bntype": "LayerNorm",   # 对应源码 354 行，可选 'LayerNorm' 或 'BatchNorm'
    "info_type": "exp",      # 对应源码 356 行，控制 inference 层的结构
    
    # --- 轨迹迭代相关参数 (forward 方法中会用到) ---
    "iter_num": 1000,        # 对应源码 392 行，SCC 更新频率或迭代次数
    "iter_type": "SCC",      # 对应源码 393 行，可选 "SCC" 或 "Gradient"
    
    # --- 训练逻辑相关参数 (Trainer 中会用到) ---
    "update_interval": 10,   # 进度条更新间隔
    "cutof": 0.5,            # Stage 1 训练提前停止的阈值系数
}

# ==================== 3. Stage 1 & Stage 2 训练 ====================
# 检查是否为稀疏矩阵，如果是则转换为稠密数组
from scipy.sparse import issparse

if issparse(st_data_use.X):
    st_data_use.X = st_data_use.X.toarray()
    
# 初始化并训练特征提取层
trainer = dt.CellDAG_Trainer(args, cdata, st_data_use)
trainer.init_train()
trainer.train_stage1(f"{save_path}/model_stage1.pkl")

# 自动寻找起始点：基于 Stage 1 嵌入进行聚类
model = torch.load(f"{save_path}/model_stage1.pkl")
model.eval()
emb = model.get_emb(isall=False)
emb_adata = sc.AnnData(emb, obs=st_data_use.obs)
sc.pp.neighbors(emb_adata, n_neighbors=15)
sc.tl.leiden(emb_adata, resolution=0.2)
st_data_use.obs['emb_cluster'] = emb_adata.obs['leiden'].values

# 设置起点：假设聚类 '0' 为发育源头
# 建议运行到此处先画个图确认下 '0' 簇在空间上的位置
start_flag = (st_data_use.obs['emb_cluster'] == '0').values
trainer.set_start_region(start_flag)

# 轨迹推断训练
trainer.train_stage2(save_path, "CD31_Path")
trainer.get_Trajectory_Ptime(knn=30, grid_num=50)

# ==================== 4. 空间拟时序与向量场可视化 ====================
# 绘制空间拟时序
dt.plot_spatial_complex(
    cdata, st_data_use, mode="time",
    value=st_data_use.obs['ptime'], title="Pseudotime (CD31 Filtered)", 
    pointsize=10, savename=f"{save_path}/Spatial_Pseudotime.pdf"
)

# 绘制轨迹向量场
xy = st_data_use.obsm['spatial_cropped_150_buffer']
plt.figure(figsize=(8, 8))
plt.scatter(xy[:, 0], xy[:, 1], c=st_data_use.obs['ptime'], cmap='Spectral_r', s=8, alpha=0.4)
plt.quiver(st_data_use.uns['E_grid'][0], st_data_use.uns['E_grid'][1], 
           st_data_use.uns['V_grid'][0], st_data_use.uns['V_grid'][1], color='black', scale=0.5)
plt.savefig(f"{save_path}/Vector_Field_Trajectory.png", dpi=1000)
plt.close()

print(f"🚀轨迹分析已完成！请查看保存路径：{save_path}")

