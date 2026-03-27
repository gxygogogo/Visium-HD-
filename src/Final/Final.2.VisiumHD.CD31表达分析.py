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
#import warnings
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")
plt.set_loglevel('WARNING')



#%% bin2cell Step1. 数据加载
sample_name = "Visium_HD_Human_Colon_Cancer_P2"
base_dir = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample_name}"
#os.makedirs(f"{base_dir}/stardist", exist_ok=True)

path = f"{base_dir}/binned_outputs/square_002um/"
#the image you used for --image of spaceranger, that's the one the spatial coordinates are based on
source_image_path = f"{base_dir}/{sample_name}_tissue_image.btf"
spaceranger_image_path = f"{base_dir}/spatial"

#%% 读入数据
cdata = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.h5ad")
# c2 = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.v2.h5ad")
# c3 = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.v3.h5ad")

# cdata.obs['Flex_CellSubType'] = c3.obs['Flex_CellSubType']
# cdata.obs['Flex_CellSubType_0.6'] = c2.obs['Flex_CellSubType_0.6']
# cdata.obs['Flex_Score'] = c2.obs['Flex_Score']
# 
# cdata.write(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.final.h5ad")

#%% 基因表达
# 1. 提取基因表达量 (确保从原始计数或标准化后的数据中提取)
cd31_expr = cdata.obs_vector("PECAM1")
dll4_expr = cdata.obs_vector("DLL4")
FLT4_expr = cdata.obs_vector("FLT4")
PIEZO1_expr = cdata.obs_vector("PIEZO1")
PIEZO2_expr = cdata.obs_vector("PIEZO2")
TRPV4_expr = cdata.obs_vector("TRPV4")
TRPC1_expr = cdata.obs_vector("TRPC1")
TRPC6_expr = cdata.obs_vector("TRPC6")

# 2. 定义“表达”的阈值 (最简单是 > 0，或者根据分布设为 p95 以上)
# 这里先用最通用的 > 0 逻辑
is_cd31_pos = cd31_expr > 0  # 也可以直接用 > 0
is_dll4_pos = dll4_expr > 0
is_flt4_pos = FLT4_expr > 0
is_piezo1_pos = PIEZO1_expr > 0
is_piezo2_pos = PIEZO2_expr > 0
is_trpv4_pos = TRPV4_expr > 0
is_trpc1_pos = TRPC1_expr > 0
is_trpc6_pos = TRPC6_expr > 0

# 与CD31共表达的比例
is_cd31_flt4_pos = is_cd31_pos & is_flt4_pos
is_cd31_piezo1_pos = is_cd31_pos & is_piezo1_pos
is_cd31_piezo2_pos = is_cd31_pos & is_piezo2_pos
is_cd31_trpv4_pos = is_cd31_pos & is_trpv4_pos
is_cd31_trpc1_pos = is_cd31_pos & is_trpc1_pos
is_cd31_trpc6_pos = is_cd31_pos & is_trpc6_pos


tip_cell_mask = (cdata.obs['predicted_labels'] == 'Tip-like ECs') & (cdata.obs['conf_score'] >= 0.6) # 只考虑高置信度的 Tip-like ECs
is_double_pos = is_cd31_pos & is_dll4_pos
num_tip_total = tip_cell_mask.sum()
num_tip_double_pos = (tip_cell_mask & is_double_pos).sum()


## 1. 计算CD31+&DLL4+在DLL4+中的比例和Tip cell中的比例
is_cd31_dll4_pos = is_cd31_pos & is_dll4_pos
num_cd31_pos = is_cd31_pos.sum()
num_dll4_pos = is_dll4_pos.sum()
num_cd31_dll4_pos = is_cd31_dll4_pos.sum()

num_tip_double_pos = (tip_cell_mask & is_cd31_dll4_pos).sum()

cdata.obs['is_cd31_dll4_pos'] = cdata.obs['predicted_labels'].astype(str) # 确保是字符串类型，避免后续赋值问题
cdata.obs.loc[tip_cell_mask & is_cd31_dll4_pos, 'is_cd31_dll4_pos'] = 'CD31+DLL4+ Tip-like ECs'
cdata.obs['is_cd31_dll4_pos'] = cdata.obs['is_cd31_dll4_pos'].astype('category') # 转回类别类型

## 绘图
target_types = ['Tip-like ECs', 'CMS2', 'Stalk-like ECs', 'CD31+DLL4+ Tip-like ECs', 'Mechanical Tip-like ECs']

# 创建颜色字典
my_palette = {
    'Tip-like ECs': '#00FF00',   # 亮绿
    'Stalk-like ECs': '#0080FF', # 亮蓝
    'CMS2': '#F291B5',           # 红色
    'CD31+DLL4+ Tip-like ECs': '#F2A413',    # 黄色
    'Mechanical Tip-like ECs': '#FF0000'  # 红色

}

# 2. 创建一个“空”的画布
# 尺寸可以根据条目数量调整
fig, ax = plt.subplots(figsize=(4, 6)) 
ax.axis('off') # 隐藏坐标轴

# 3. 手动创建图例句柄 (Legend Handles)
legend_elements = [
    Patch(facecolor=color, edgecolor='none', label=label) 
    for label, color in my_palette.items()
]

# 4. 在画布中心绘制图例
# loc='center': 居中显示
# frameon=False: 不要边框
# prop={'size': 14}: 调整字体大小
legend = ax.legend(
    handles=legend_elements, 
    loc='center', 
    frameon=False, 
    prop={'size': 16, 'weight': 'bold'},
    labelspacing=1.2 # 条目之间的间距
)

# 5. 高清保存为独立文件
# 保存为 PDF 矢量图，方便后期放入论文或 PPT 中无损放大
output_legend = f"{base_dir}/stardist/Independent_Legend.pdf"
plt.savefig(output_legend, bbox_inches='tight', transparent=True)

# ================= 2. 准备数据子集 =================
# 过滤掉不在列表中的细胞，或者只显示高置信度的点
confidence_threshold = 0.6
mask = (cdata.obs['is_cd31_dll4_pos'].isin(target_types)) & (cdata.obs['conf_score'] >= confidence_threshold)
adata_multi = cdata[mask].copy()

# ================= 3. 绘图 =================
fig, ax = plt.subplots(figsize=(15, 15))

# 修复缩放因子
lib_id = list(adata_multi.uns['spatial'].keys())[0]
adata_multi.uns['spatial'][lib_id]['scalefactors']['tissue_hires_scalef'] = 1.0

sc.pl.spatial(
    adata_multi,
    color='is_cd31_dll4_pos',
    img_key="0.5_mpp_150_buffer",
    basis="spatial_cropped_150_buffer",
    
    # --- 核心控制 ---
    palette=my_palette,         # 使用自定义颜色
    alpha_img=0.3,              # 进一步压暗 HE 背景，让彩色点更突出
    spot_size=40,               # 细胞多的时候点可以稍微调小一点
    
    frameon=False,
    show=False,
    ax=ax
)

ax.set_facecolor('black')

# 4. 保存
output_multi = f"{base_dir}/stardist/Tipcell_次级分类.png"
plt.savefig(output_multi, dpi=600, bbox_inches='tight', facecolor='black')
print(f"多类型叠加图已保存至: {output_multi}")


## 2. 计算CD31+与机械力受体基因双阳的比例
is_cd31_piezo1_pos = is_cd31_pos & is_piezo1_pos
num_cd31_piezo1_pos = is_cd31_piezo1_pos.sum()
num_cd31_pos = is_cd31_pos.sum()
num_piezo1_pos = is_piezo1_pos.sum()

is_cd31_piezo2_pos = is_cd31_pos & is_piezo2_pos
num_cd31_piezo2_pos = is_cd31_piezo2_pos.sum()
num_piezo2_pos = is_piezo2_pos.sum() 

is_cd31_trpv4_pos = is_cd31_pos & is_trpv4_pos
num_cd31_trpv4_pos = is_cd31_trpv4_pos.sum()
num_trpv4_pos = is_trpv4_pos.sum()

is_cd31_trpc1_pos = is_cd31_pos & is_trpc1_pos
num_cd31_trpc1_pos = is_cd31_trpc1_pos.sum()
num_trpc1_pos = is_trpc1_pos.sum()

is_cd31_trpc6_pos = is_cd31_pos & is_trpc6_pos
num_cd31_trpc6_pos = is_cd31_trpc6_pos.sum()
num_trpc6_pos = is_trpc6_pos.sum()

## 3. 计算机械力受体和tip cell的共表达比例
is_piezo1_tip_pos = is_piezo1_pos & tip_cell_mask
num_piezo1_tip_pos = is_piezo1_tip_pos.sum()
num_piezo1_pos = is_piezo1_pos.sum()

is_piezo2_tip_pos = is_piezo2_pos & tip_cell_mask
num_piezo2_tip_pos = is_piezo2_tip_pos.sum()
num_piezo2_pos = is_piezo2_pos.sum()

is_trpv4_tip_pos = is_trpv4_pos & tip_cell_mask
num_trpv4_tip_pos = is_trpv4_tip_pos.sum()
num_trpv4_pos = is_trpv4_pos.sum()

is_trpc1_tip_pos = is_trpc1_pos & tip_cell_mask
num_trpc1_tip_pos = is_trpc1_tip_pos.sum()
num_trpc1_pos = is_trpc1_pos.sum()

is_trpc6_tip_pos = is_trpc6_pos & tip_cell_mask
num_trpc6_tip_pos = is_trpc6_tip_pos.sum()
num_trpc6_pos = is_trpc6_pos.sum()

is_dll4_piezo1_pos = is_dll4_pos & is_piezo1_pos
num_dll4_piezo1_pos = is_dll4_piezo1_pos.sum()
num_piezo1_pos = is_piezo1_pos.sum()

is_dll4_piezo2_pos = is_dll4_pos & is_piezo2_pos
num_dll4_piezo2_pos = is_dll4_piezo2_pos.sum()
num_piezo2_pos = is_piezo2_pos.sum()

is_dll4_trpv4_pos = is_dll4_pos & is_trpv4_pos
num_dll4_trpv4_pos = is_dll4_trpv4_pos.sum()
num_trpv4_pos = is_trpv4_pos.sum()

is_dll4_trpc1_pos = is_dll4_pos & is_trpc1_pos
num_dll4_trpc1_pos = is_dll4_trpc1_pos.sum()
num_trpc1_pos = is_trpc1_pos.sum()

is_dll4_trpc6_pos = is_dll4_pos & is_trpc6_pos
num_dll4_trpc6_pos = is_dll4_trpc6_pos.sum()
num_trpc6_pos = is_trpc6_pos.sum()

cdata.obs['is_cd31_dll4_pos'] = cdata.obs['is_cd31_dll4_pos'].astype(str) # 确保是字符串类型，避免后续赋值问题
cdata.obs.loc[is_piezo1_tip_pos | is_piezo2_tip_pos | is_trpv4_tip_pos | is_trpc1_tip_pos | is_trpc6_tip_pos, 'is_cd31_dll4_pos'] = 'Mechanical Tip-like ECs'
cdata.obs['is_cd31_dll4_pos'] = cdata.obs['is_cd31_dll4_pos'].astype('category') # 转回类别类型


## 3. CD31为中心的Niche分析
import squidpy as sq
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ================= 1. 空间图构建 (Spatial Graph) =================
cdata.obs['predicted_labels'] = cdata.obs['predicted_labels'].astype(str) # 确保是字符串类型，避免后续赋值问题
cdata.obs.loc[cd31_expr > np.quantile(cd31_expr, 0.90),  'predicted_labels'] = 'CD31+ cells' # 将 CD31 高表达的细胞标记为 'CD31+ cells'
cdata.obs['predicted_labels'] = cdata.obs['predicted_labels'].astype('category')

# 1. 找到数量太少的类别
counts = cdata.obs['predicted_labels'].value_counts()
to_keep = counts[counts > 20].index # 过滤掉 CMS4, Enterocytes type 1 等

# 2. 创建一个子集
cdata_clean = cdata[cdata.obs['predicted_labels'].isin(to_keep)].copy()
cdata_clean.obs['predicted_labels'] = cdata_clean.obs['predicted_labels'].cat.remove_unused_categories()
sq.gr.spatial_neighbors(cdata_clean, radius=100, coord_type="generic", spatial_key="spatial_cropped_150_buffer")

# 3. 重新计算和绘图
sq.gr.nhood_enrichment(cdata_clean, cluster_key="predicted_labels")

# 去除对角线
#enrich_data = cdata_clean.uns["new_labels_nhood_enrichment"]
#
#if isinstance(enrich_data, dict):
#    # 根据你提供的输出，这里应该修改 'zscore'
#    # 如果你想修改计数，也可以对 'count' 做同样的操作
#    mat = enrich_data['zscore'].copy() 
#    np.fill_diagonal(mat, 0)
#    cdata_clean.uns["new_labels_nhood_enrichment"]['zscore'] = mat
#else:
#    # 兼容旧版本数组格式
#    mat = enrich_data.copy()
#    np.fill_diagonal(mat, 0)
#    cdata_clean.uns["new_labels_nhood_enrichment"] = mat

sq.pl.nhood_enrichment(cdata_clean, 
                       cluster_key="predicted_labels", 
                       method="single", 
                       cmap="RdBu_r",
                       vmax = 100,
                       vmin = -100,)
plt.savefig(f"{base_dir}/stardist/Global_Vascular_Niche_old.png", dpi = 600, bbox_inches = 'tight')
plt.close()






import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 区域聚类模块 (Panel 1: Spatial Niche Domains) =================
print("正在构建空间 Domain...")
# 确保邻接矩阵已构建 (radius 决定了 Domain 的平滑度)
sq.gr.spatial_neighbors(cdata_clean, 
                        radius=100, 
                        coord_type="generic", 
                        spatial_key="spatial_cropped_150_buffer")

adj = cdata_clean.obsp["spatial_connectivities"]
node_labels = pd.get_dummies(cdata_clean.obs['predicted_labels'])
nhood_counts = adj @ node_labels.values
row_sums = nhood_counts.sum(axis=1)
row_sums[row_sums == 0] = 1
nhood_prop = nhood_counts / row_sums[:, None]

# 聚类获得大块 Domain
niche_adata = sc.AnnData(X=nhood_prop, obs=cdata_clean.obs.copy())
sc.pp.pca(niche_adata, n_comps=20)
sc.pp.neighbors(niche_adata, n_neighbors=20)
# resolution=0.1 是获得大块连通区域的关键
sc.tl.leiden(niche_adata, 
             resolution=0.00001, 
             key_added="spatial_niche")
cdata_clean.obs['spatial_niche'] = niche_adata.obs['spatial_niche'].astype('category')

# 可视化 Panel 1
sc.pl.spatial(cdata_clean, 
              color='spatial_niche', 
              spot_size=30, 
              #basis="spatial_cropped_150_buffer", 
              title="Panel 1: Spatial Niche Domains", 
              frameon=False)
plt.savefig(f"{base_dir}/stardist/Panel1_Spatial_Niche_Domains.png", dpi=600, bbox_inches='tight')
plt.close()

# ================= 2. 成分统计模块 (Panel 2: Cell Type Proportions) =================
print("正在计算 Niche 内部成分...")
# 计算交叉表
ct = pd.crosstab(cdata_clean.obs['spatial_niche'], cdata_clean.obs['new_labels'], normalize='index')

# 绘制堆叠柱状图 (或饼图数据)
ct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20').legend(bbox_to_anchor=(1, 1))
plt.title("Panel 2: Cell Type Composition per Niche")
plt.ylabel("Proportion")
plt.show()

# ================= 3. 目标细胞分布 (Panel 3: Vascular Cell Distribution) =================
print("正在分析血管细胞的空间偏好...")
# 筛选你关心的核心细胞类型
target_cells = ['CD31+ cells', 'Tip-like ECs', 'Stalk-like ECs']
# 计算这些细胞在不同 Niche 中的分布百分比
subset_ct = pd.crosstab(cdata_clean.obs['new_labels'], cdata_clean.obs['spatial_niche'], normalize='index')
vascular_dist = subset_ct.loc[subset_ct.index.isin(target_cells)]

# 可视化 Panel 3
vascular_dist.T.plot(kind='bar', figsize=(8, 5))
plt.title("Panel 3: Distribution of Vascular Subtypes across Niches")
plt.ylabel("Fraction of Cell Type")
plt.show()

# ================= 4. 功能分子关联 (Panel 4: Mechanosensing / PIEZO1) =================
print("正在关联功能分子表达...")
# 以 PIEZO1 为例，计算每个 Niche 的平均表达量
# 假设 PIEZO1 在 cdata_clean.X 中
cdata_clean.obs['PIEZO1_exp'] = cdata_clean[:, 'PIEZO1'].X.toarray().flatten()

plt.figure(figsize=(8, 5))
sns.barplot(data=cdata_clean.obs, x='spatial_niche', y='PIEZO1_exp', ci=68)
plt.title("Panel 4: PIEZO1 Expression Specificity per Niche")
plt.show()
















# # 建议 radius 设为 50-100um，捕捉周围 3-5 层细胞的 niche
# sq.gr.spatial_neighbors(cdata, radius=100, coord_type="generic", spatial_key="spatial_cropped_150_buffer")
# 
# # ================= 2. 方案一：全局血管 (CD31+) Niche 分析 =================
# # 我们查看所有内皮细胞（Endothelial cells）与其他细胞的交互
# sq.gr.nhood_enrichment(cdata, cluster_key="predicted_labels")
# 
# # 可视化全局血管的社交圈
# print("绘制全局血管 Niche 富集热图...")
# sq.pl.nhood_enrichment(cdata, 
#                        cluster_key = "predicted_labels",  
#                        method = "single", 
#                        cmap = "RdBu_r", 
#                        title = "Global Niche")
# plt.savefig(f"{base_dir}/stardist/Global_Vascular_Niche.png", dpi = 600, bbox_inches = 'tight')
# plt.close()


# ================= 3. 方案二：Tip Cell 特异性 Niche 及差异分析 =================
# 使用你之前标记好的 'is_cd31_dll4_pos' 列，其中包含 'CD31+DLL4+ Tip-like ECs'
sq.gr.nhood_enrichment(cdata, cluster_key="is_cd31_dll4_pos")

# 提取富集矩阵进行差异计算
nhood_res = cdata.uns["is_cd31_dll4_pos_nhood_enrichment"]
# 目标：对比 'CD31+DLL4+ Tip-like ECs' vs 'Endothelial cells' (普通内皮)
tip_enrich = nhood_res[cdata.obs['is_cd31_dll4_pos'].cat.categories.get_loc('CD31+DLL4+ Tip-like ECs')]
stalk_enrich = nhood_res[cdata.obs['is_cd31_dll4_pos'].cat.categories.get_loc('Endothelial cells')]

# 计算 Niche 差异得分
niche_diff = pd.DataFrame({
    'Cell_Type': cdata.obs['is_cd31_dll4_pos'].cat.categories,
    'Diff_Score': tip_enrich - stalk_enrich
}).sort_values(by='Diff_Score', ascending=False)

# 可视化 Tip cell 的特异性邻居
plt.figure(figsize=(10, 6))
sns.barplot(data=niche_diff.head(10), x='Diff_Score', y='Cell_Type', palette='Reds_r')
plt.title("Niche Preference: Tip Cells vs Stalk Cells")
plt.xlabel("Enrichment Difference (Tip > Stalk)")
plt.show()

# ================= 4. 空间共出现分析 (Co-occurrence) =================
# 看看随着距离增加，肿瘤细胞或成纤维细胞出现的概率
sq.gr.co_occurrence(cdata, cluster_key="is_cd31_dll4_pos")

# 专门看 Tip Cell 周围的邻居随距离分布的变化
sq.pl.co_occurrence(
    cdata, 
    cluster_key="is_cd31_dll4_pos", 
    clusters="CD31+DLL4+ Tip-like ECs",
    figsize=(12, 4)
)


## 4. 内皮细胞的分化轨迹


