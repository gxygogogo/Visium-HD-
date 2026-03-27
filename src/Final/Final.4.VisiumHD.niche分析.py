import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from scipy import sparse
import anndata
import os
import matplotlib.pyplot as plt
import pickle
from scipy.io import mmwrite
import skmisc
import scniche as sn
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

# P2: 6
# P5: 4



#%% bin2cell Step1. 数据加载
sample_name = "Visium_HD_Human_Colon_Cancer_P2"
base_dir = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample_name}"
#os.makedirs(f"{base_dir}/stardist", exist_ok=True)

path = f"{base_dir}/binned_outputs/square_002um/"
#the image you used for --image of spaceranger, that's the one the spatial coordinates are based on
source_image_path = f"{base_dir}/{sample_name}_tissue_image.btf"
spaceranger_image_path = f"{base_dir}/spatial"

cdata = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.final.h5ad")
cdata = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_scNiche_adata.h5ad")

#%% scniche
# 1. 将细胞标签转为 One-hot 编码 (类似反卷积后的丰度)
decon_res = pd.get_dummies(cdata.obs['predicted_labels']).astype(float)
cdata.obsm['X_C2L'] = decon_res.values
cdata.uns['CN_order'] = decon_res.columns.tolist()

# 2. 检查是否有 PCA 降维，没有的话跑一下 (scNiche 需要低维表达特征)
if 'X_pca' not in cdata.obsm:
    sc.pp.pca(cdata, n_comps=30)

# 1. 强制给所有细胞分配同一个样本 ID（如果你分析的是单张切片）
cdata.obs['sample_id'] = 'Visium_P5'

# 2. 如果你之前用的是 'object_id'，检查一下它的分布
# 如果 object_id 很多且每个只有几个细胞，那是不对的
print(cdata.obs['object_id'].value_counts().head())

# 3. 确保坐标列命名符合 scNiche 的预期
# scNiche 内部有时会寻找 'spatial' 或 'x', 'y'。
# 我们把坐标提取出来放入 obs 确保万无一失
cdata.obs['x_new'] = cdata.obsm['spatial_cropped_150_buffer'][:, 0]
cdata.obs['y_new'] = cdata.obsm['spatial_cropped_150_buffer'][:, 1]

# 重新运行，注意 sample_key 的变化
cdata = sn.pp.process_multi_slices(
    adata=cdata,
    celltype_key='predicted_labels',
    sample_key='sample_id', # 使用统一的 ID
    mode='KNN',
    k_cutoff=4,
    is_pca=True,
    verbose=True,
    layer_key='X_pca'
)

# 准备批次数据
choose_views = ['X_C2L', 'X_data', 'X_data_nbr']
cdata = sn.pp.prepare_data_batch(adata=cdata, batch_num=16, choose_views=choose_views)

# training
model = sn.tr.Runner_batch(adata=cdata, device='cuda:0', verbose=False, choose_views=choose_views)
cdata = model.fit(lr=0.01, epochs=30)

# 4. Niche 聚类
# target_k 是你想要的 Niche 数量，建议根据组织结构设为 5-10
target_k = 15
cdata = sn.tr.clustering(cdata, target_k=target_k)

# 5. 可视化
sc.pl.spatial(
    cdata,
    color='scNiche',
    spot_size=30,
    title="Spatial Niche Domains",
    frameon=False,
    show=False
)
plt.savefig(f"{base_dir}/stardist/scNiche_niche_spatial_6_{target_k}.pdf", dpi=1000, bbox_inches='tight')
plt.close()


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=15, random_state=42)
cdata.obs['scNiche_gmm'] = gmm.fit_predict(cdata.obsm['X_scniche']).astype(str)

# 2. 获取聚类的置信度 (Log-likelihood)，数值越高代表该点越符合所属簇的分布
cdata.obs['gmm_confidence'] = gmm.score_samples(cdata.obsm['X_scniche'])

# 3. 绘图对比
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

# 图 A：UMAP 上的分类结果
sc.pl.umap(
    cdata, color='scNiche_gmm', ax=ax1, show=False,
    title='GMM Clustering (UMAP)', legend_loc='on data'
)

# 图 B：空间分布图 (分类结果)
sc.pl.embedding(
    cdata, basis="spatial_cropped_150_buffer", 
    color='scNiche_gmm', ax=ax2, show=False,
    title='GMM Spatial Map'
)

# 图 C：空间置信度图 (识别过渡带)
sc.pl.embedding(
    cdata, basis="spatial_cropped_150_buffer", 
    color='gmm_confidence', cmap='viridis', ax=ax3, show=False,
    title='GMM Confidence (Yellow = High)'
)

plt.tight_layout()
plt.savefig(f"{base_dir}/stardist/GMM_Niche_Analysis.png", dpi=300)
plt.show()

sc.pl.spatial(
    cdata,
    color='scNiche_gmm',
    spot_size=30,
    title="Spatial Niche Domains",
    frameon=False,
    show=False
)
plt.savefig(f"{base_dir}/stardist/GMM/scNiche_N_6_niche_spatial.pdf", dpi=1000, bbox_inches='tight')
plt.close()



# UMAP 
# 1. 在 scNiche 特征上构建邻居图
# 默认 scNiche 的特征通常存在 obsm['X_scniche'] 中
sc.pp.neighbors(cdata, use_rep='X_scniche', n_neighbors=80, key_added='niche_neighbors')

# 2. 跑 UMAP
sc.tl.umap(cdata, neighbors_key='niche_neighbors', min_dist = 0.8)

# 3. 绘制
with plt.rc_context({'figure.figsize': (10, 8)}):
    sc.pl.umap(
        cdata, 
        color=['scNiche', 'predicted_labels'], # 同时对比 Niche 和细胞类型
        ncols=2,
        size=5, 
        palette=cdata.uns['scNiche_colors'], # 使用你刚才自动获取的颜色
        frameon=False,
        show=False
    )
plt.savefig(f"{base_dir}/stardist/scNiche_Specific_UMAP.pdf", dpi=800, bbox_inches='tight')





# 6. 富集分析 (找出 Tip-like ECs 的家)
# 检查每个 Niche 的细胞数量
print(cdata.obs['scNiche_gmm'].value_counts())

# 如果有非常小的 Niche (比如小于 10 个细胞)，建议先过滤掉
niche_counts = cdata.obs['scNiche_gmm'].value_counts()
valid_niches = niche_counts[niche_counts > 20].index # 阈值设为 20 或更高
cdata_sub = cdata[cdata.obs['scNiche_gmm'].isin(valid_niches)].copy()
#sn.al.enrichment_spot(adata=cdata, deconvoluted_res=decon_res, val_key='scNiche')
sn.al.enrichment(
    adata=cdata,
    id_key='predicted_labels',   # 细胞类型
    val_key='scNiche_gmm'    # niche cluster
)

# 绘制热图
sn.pl.enrichment_heatmap(
    adata=cdata,
    id_key='predicted_labels',
    val_key='scNiche_gmm',
    kwargs={'vmin': -4, 'vmax': 4, 'cmap': 'RdBu_r', 'figsize': (14, 8)}
)
plt.savefig(f"{base_dir}/stardist/GMM/scNiche_Enrichment.pdf", dpi=1000, bbox_inches='tight')
plt.close()

# # 1. 检查并删除 uns 中无法保存的 dataloader 对象
# if 'dataloader' in cdata.uns:
#     del cdata.uns['dataloader']
#     print("已删除无法保存的 dataloader 对象。")
# 
# # 2. 有些版本的 scNiche 还会把 model 存进去，也建议清理掉
# if 'model' in cdata.uns:
#     del cdata.uns['model']
# 
# # 3. 现在重新执行保存
# save_path = f"{base_dir}/stardist/GMM/{sample_name}_scNiche_Nibour_4_GMM_15_adata.h5ad"
# cdata.write(save_path)

# 单独绘制每个niche
all_niches = sorted(cdata.obs['scNiche_gmm'].unique().astype(str))

# 3. 循环绘图
for niche_id in all_niches:
    print(f"正在绘制 Niche {niche_id}...")
    
    # 使用 rc_context 统一设置背景颜色和画布大小
    with plt.rc_context({'axes.facecolor': 'black', 'figure.figsize': (10, 10)}):
        sc.pl.spatial(
            cdata,
            color='scNiche_gmm',
            groups=[niche_id],           # 核心：只显示当前这个 Niche
            basis="spatial_cropped_150_buffer", # 关键：使用局部对齐坐标
            library_id=None,             # 如果报错，请填入 cdata.uns['spatial'] 里的 key
            img_key='0.5_mpp_150_buffer',             # 叠加高分辨率 HE 图像
            alpha_img=0.5,               # HE 图透明度，方便看清 Spot 颜色
            spot_size=35,                # 根据你的切片密集程度微调
            title=f"Spatial Distribution: Niche {niche_id}",
            frameon=False,
            show=False
        )
        
        # 保存为单独的文件
        plt.savefig(f"{base_dir}/stardist/GMM/Niche_{niche_id}_on_HE.pdf", dpi=300, bbox_inches='tight')
        plt.close()



cdata_raw = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.h5ad")
cdata.uns['spatial'] = cdata_raw.uns['spatial']  # 确保 spatial 信息完整
# 获取 library_id 和 所有的 Niche 标签
lib_id = list(cdata.uns['spatial'].keys())[0]
all_niches = cdata.obs['scNiche'].cat.categories.tolist()

# 核心修复：确保缩放因子一致
cdata.uns['spatial'][lib_id]['scalefactors']['tissue_hires_scalef'] = 1.0

# 2. 自动获取 scNiche 的颜色列表
# Scanpy 默认将颜色存在 uns['{cluster_key}_colors'] 中
niche_color_list = cdata.uns['scNiche_colors']
# 建立 Niche 名字到颜色的映射字典
auto_palette = dict(zip(all_niches, niche_color_list))

print(f"检测到 {len(all_niches)} 个 Niche，已自动加载颜色映射。")

# 3. 循环绘制每个 Niche
for niche_id in all_niches:
    print(f"正在绘制 Niche {niche_id} (颜色: {auto_palette[niche_id]})...")
    
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # 绘图逻辑
    sc.pl.spatial(
        cdata,
        color='scNiche',
        groups=[niche_id],              # 只高亮当前 Niche
        palette=auto_palette,           # 使用自动获取的全量色板
        basis="spatial_cropped_150_buffer",
        img_key="0.5_mpp_150_buffer",   # 叠加 HE
        
        # --- 视觉控制 ---
        alpha_img=0.3,                  # 压暗背景
        spot_size=40,                   # 点的大小
        frameon=False,
        show=False,
        ax=ax,
        legend_loc=None                 # 单图不需要图例
    )
    
    # 设置黑色底色，让 Niche 颜色更显眼
    ax.set_facecolor('black')
    ax.set_title(f"Spatial Niche: {niche_id}", color='white', fontsize=24, pad=20)

    # 保存
    output_path = f"{base_dir}/stardist/Niche_{niche_id}_on_HE2.pdf"
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='black')
    plt.close()



# 7. 提取 Tip-like ECs 细胞子集
cell_names = ['Stalk-like ECs', 'Tip-like ECs']  # 根据你的预测标签修改
for cell_name in cell_names:
    tip_cells = cdata.obs[cdata.obs['predicted_labels'] == cell_name].copy()

    # 2. 计算 Tip-like ECs 在各个 Niche 中的分布百分比
    # 注意：这里计算的是 (该 Niche 内的 Tip 数量 / 全片 Tip 总量) * 100
    tip_dist = tip_cells['scNiche_gmm'].value_counts(normalize=True).sort_index() * 100

    # 3. 绘图
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(tip_dist)) # 或者使用你之前 Niche 的专用色板

    ax = sns.barplot(x=tip_dist.index, y=tip_dist.values, palette=colors, edgecolor='black')

    # 装饰
    plt.title(f'Distribution of {cell_name} across Niches', fontsize=15, pad=15)
    plt.xlabel('scNiche Cluster', fontsize=12)
    plt.ylabel(f'% of Total {cell_name}', fontsize=12)

    # 在柱子上标注具体数值
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10)

    plt.ylim(0, max(tip_dist.values) + 10) # 留出顶部空间给数值标注
    plt.xticks(rotation = 45)
    plt.savefig(f"{base_dir}/stardist/GMM/{cell_name}_Niche_Distribution.pdf", bbox_inches='tight')
    plt.close()




# 8. 某个基因在各个niche中的分布
import scanpy as sc
import matplotlib.pyplot as plt

# 1. 指定你关注的血管/肿瘤相关基因
# 示例：PECAM1(内皮), VEGFA(促血管), ESM1(Tip细胞标志), COL1A1(基质)
target_genes = ['PECAM1', 'DLL4', 'ESM1', 'EST1', "APLN", 'ANGPT2', 'FLT4', 'LYVE1', 'VEGFA', 'COL1A1', 'PIEZO1', 'PIEZO2', 'TRPV4', 'TRPC1', 'TRPC6'] 

# 检查基因是否在数据中
genes_to_plot = [g for g in target_genes if g in cdata.var_names]

# 2. 绘制小提琴图
sc.pl.violin(
    cdata, 
    keys=genes_to_plot, 
    groupby='scNiche', 
    rotation=0, 
    stripplot=False,  # 点太多，关掉 stripplot 否则会糊成一团
    inner='box',      # 显示箱线图
    show=False
)
plt.suptitle("Gene Expression Distribution across scNiches", fontsize=15)
plt.savefig(f"{base_dir}/stardist/scNiche_Gene_Violin.pdf", bbox_inches='tight')





# 1. 设置基因列表（建议按功能逻辑分组，这样图上方会有标签）
var_names_raw = {
    'Tip like': ['PECAM1', 'DLL4', 'ESM1', 'ETS1', "APLN", 'ANGPT2', 'FLT4', 'LYVE1', 'VEGFA'],
    'Mechano': ['PIEZO1', 'PIEZO2', 'TRPV4', 'TRPC1', 'TRPC6']
}

# 2. 自动筛选出数据中真实存在的基因，防止 KeyError
var_names = {}
for group, genes in var_names_raw.items():
    # 只保留存在于 cdata.var_names 中的基因
    existing_genes = [g for g in genes if g in cdata.var_names]
    if existing_genes:
        var_names[group] = existing_genes
    else:
        print(f"警告：组 {group} 中的所有基因在数据中都不存在！")

# 打印一下没找到的基因，方便你核对
all_input_genes = [g for genes in var_names_raw.values() for g in genes]
missing_genes = set(all_input_genes) - set(cdata.var_names)
if missing_genes:
    print(f"❌ 以下基因未在 cdata 中找到，已被跳过: {missing_genes}")

# 2. 绘图
# 注意：fraction_threshold 是让小点“消失”的关键
with plt.rc_context({'figure.figsize': (14, 6)}):
    ax = sc.pl.dotplot(
        cdata, 
        var_names=var_names, 
        groupby='scNiche_gmm', 
        standard_scale='var',      # 按基因标准化 (0-1)，突出特异性
        cmap='magma',               # 经典红色系
        var_group_rotation=0,
        # --- 核心规范化参数 ---
        #dot_min=None,               # 最小点显示的物理尺寸比例
        #dot_max=2,               # 最大点显示的物理尺寸比例（防止遮挡邻近格子）
        smallest_dot=3,
        # --- 过滤噪音小点 ---
        #fraction_threshold=0.1,    # 只有在该 Niche 中表达细胞占比 > 10% 才会出点
        # --- 视觉微调 ---
        title='Normalized Gene Expression per Niche',
        colorbar_title='Relative Expression',
        size_title='Fraction of Cells',
        #stripplot=False,
        show=False
    )

# 3. 细节美化：旋转横轴标签防止重叠
#plt.xticks(rotation=45, ha='right')

# 4. 保存
output_path = f"{base_dir}/stardist/GMM/scNiche_Gene_Dotplot_Clean.pdf"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()








for i in all_niches:
    # 2. Extract subset of cells belonging to Niche 9
    try:
        niche_9_cells = cdata.obs[cdata.obs['scNiche_gmm'] == i]
    except KeyError:
        print(f"Error: Niche '{i}' not found in cdata.obs['scNiche_gmm']. Check your clusters.")
        # Exits the script to prevent crashing below
        raise

    if niche_9_cells.empty:
        print(f"Warning: Niche {i} is empty. No cells to plot.")
    else:
        # 3. Calculate value counts of cell types (predicted_labels)
        type_counts = niche_9_cells['predicted_labels'].value_counts()

        # 4. Filter for clarity: Group rare cell types into "Others"
        # threshold = 0.02 (2% or less) is generally a good cutoff for 20+ cell types
        threshold = 0.02 
        total_cells = len(niche_9_cells)

        # Identify types above threshold
        mask = (type_counts / total_cells) > threshold
        main_types = type_counts[mask]

        # Create the "Others" slice if there are filtered types
        others_count = type_counts[~mask].sum()
        if others_count > 0:
            main_types['Others'] = others_count

        # ==================== Visualization ====================
        # 5. Set up plot
        # Use a professional palette (e.g., 'tab20c' or 'deep') suitable for many types
        colors = sns.color_palette('tab20', len(main_types))

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(aspect="equal"))

        # 6. Create Pie Chart
        # autopilot=['%1.1f%%']: adds percentage labels
        # pctdistance=0.85: pulls percentages inward for better reading near edges
        # textprops={'color':"w"}: forces percentage text to be white for contrast
        # startangle=90: rotates chart for traditional top-start
        # wedgeprops: adds thin white border between slices
        wedges, texts, autotexts = ax.pie(
            main_types.values, 
            autopct='%1.1f%%',
            pctdistance=0.85, 
            textprops=dict(color="w", weight="bold"),
            startangle=90, 
            colors=colors,
            wedgeprops={"edgecolor":"w",'linewidth': 1, 'antialiased': True}
        )

        # 7. Add Legend on the right
        # ha='center', va='center': centers the entire legend box vertically
        # ncol=2: if you have many types, uses two columns for legend
        ax.legend(
            wedges, 
            main_types.index,
            title="Cell Types",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=12,
            title_fontsize=14
        )

        # 8. Aesthetics and Title
        plt.setp(autotexts, size=11, weight="bold")
        ax.set_title(f"Cellular Composition: Spatial Niche{i}", fontsize=18, weight="bold")
        plt.tight_layout()

        # 9. Save as PDF for vector graphics (paper quality)
        output_path = f"{base_dir}/stardist/GMM/Niche_{i}_Composition_Pie.pdf"
        plt.savefig(output_path, dpi=800, bbox_inches='tight')



































