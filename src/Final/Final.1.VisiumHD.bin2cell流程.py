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
warnings.filterwarnings("ignore")
plt.set_loglevel('WARNING')



#%% bin2cell Step1. 数据加载
sample_name = "Visium_HD_Human_Colon_Cancer_P5"
base_dir = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample_name}"
#os.makedirs(f"{base_dir}/stardist", exist_ok=True)

path = f"{base_dir}/binned_outputs/square_002um/"
#the image you used for --image of spaceranger, that's the one the spatial coordinates are based on
source_image_path = f"{base_dir}/{sample_name}_tissue_image.btf"
spaceranger_image_path = f"{base_dir}/spatial"

'''
保将高分辨率 H&E 图像、2μm 的 Bin 计数矩阵以及缩放因子(Scale factors)正确关联并载入 AnnData 对象。
'''
adata = b2c.read_visium(path, 
                        source_image_path = source_image_path, 
                        spaceranger_image_path = spaceranger_image_path
                       )
adata.var_names_make_unique()
adata

# 过滤掉表达极低的基因和没有任何UMI的bin
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_counts=1)
adata


#%% bin2cell Step2. 图像与处理，去除条纹，生成细胞标签
mpp = 0.5
b2c.scaled_he_image(adata, mpp=mpp, save_path=f"{base_dir}/stardist/he.tiff")

# 去除Visium HD芯片横向和纵向捕获的信号差异
b2c.destripe(adata)


#define a mask to easily pull out this region of the object in the future
mask = ((adata.obs['array_row'] >= 2050) & 
        (adata.obs['array_row'] <= 2250) & 
        (adata.obs['array_col'] >= 1350) & 
        (adata.obs['array_col'] <= 1550)
       )

bdata = adata[mask]
sc.pl.spatial(bdata, 
              color=[None, "n_counts", "n_counts_adjusted"], 
              color_map="OrRd", 
              img_key="0.5_mpp_150_buffer", 
              basis="spatial_cropped_150_buffer")
plt.savefig(f"{base_dir}/stardist/spatial_qc.png", bbox_inches='tight', dpi=300)
plt.close() # 及时关闭画布，防止内存占用

sc.pl.spatial(bdata, 
              color=[None, "n_counts", "n_counts_adjusted"], 
              color_map="OrRd")
plt.savefig(f"{base_dir}/stardist/spatial_qc_no_img.png", bbox_inches='tight', dpi=300)
plt.close()


#%% bin2cell Step3. 基于H&E进行细胞核分割
b2c.stardist(image_path=f"{base_dir}/stardist/he.tiff", 
             labels_npz_path=f"{base_dir}/stardist/he.npz", 
             stardist_model="2D_versatile_he", 
             prob_thresh=0.01
            )

'''
将图像上的分割标签(Label ID)映射回 AnnData。它计算每个 2μm Bin 落在图像的哪个像素上，并赋予对应的 Label ID。
'''
b2c.insert_labels(adata, 
                  labels_npz_path=f"{base_dir}/stardist/he.npz", 
                  basis="spatial", 
                  spatial_key="spatial_cropped_150_buffer",
                  mpp=mpp, 
                  labels_key="labels_he"
                 )


bdata = adata[mask]
#the labels obs are integers, 0 means unassigned
bdata = bdata[bdata.obs['labels_he']>0]
bdata.obs['labels_he'] = bdata.obs['labels_he'].astype(str)
sc.pl.spatial(bdata, color=[None, "labels_he"], img_key="0.5_mpp_150_buffer", basis="spatial_cropped_150_buffer")
plt.savefig(f"{base_dir}/stardist/insert_labels.png", bbox_inches='tight', dpi=300)
plt.close()



#the label viewer wants a crop of the processed image
#get the corresponding coordinates spanning the subset object
crop = b2c.get_crop(bdata, basis="spatial", spatial_key="spatial_cropped_150_buffer", mpp=mpp)
rendered = b2c.view_labels(image_path=f"{base_dir}/stardist/he.tiff", 
                           labels_npz_path=f"{base_dir}/stardist/he.npz", 
                           crop=crop
                          )
plt.imshow(rendered)
plt.savefig(f"{base_dir}/stardist/rendered.png", bbox_inches='tight', dpi=300)
plt.close()


'''
由于 H&E 分割只识别核，此函数根据设置（如扩张 2 个 Bin) 或体积比例算法，将标签向外扩张，覆盖细胞质区域。
'''
b2c.expand_labels(adata, 
                  labels_key='labels_he', 
                  expanded_labels_key="labels_he_expanded"
                 )


bdata = adata[mask]
#the labels obs are integers, 0 means unassigned
bdata = bdata[bdata.obs['labels_he_expanded']>0]
bdata.obs['labels_he_expanded'] = bdata.obs['labels_he_expanded'].astype(str)
sc.pl.spatial(bdata, color=[None, "labels_he_expanded"], img_key="0.5_mpp_150_buffer", basis="spatial_cropped_150_buffer")
plt.savefig(f"{base_dir}/stardist/label_expanded.png", bbox_inches='tight', dpi=300)
plt.close()


#%% bin2cell Step4. 基于基因表达(GEX)辅助进行分割
# 将校正后的总 UMI 数转化成一张“热力图”样式的 TIFF。这本质上是把基因数据当成一张荧光图像。
b2c.grid_image(adata, "n_counts_adjusted", mpp=mpp, sigma=5, save_path=f"{base_dir}/stardist/gex.tiff")

# 使用 StarDist 的荧光模型（2D_versatile_fluo）对上面的热力图进行分割。这种方法在组织稀疏区效果极佳。
b2c.stardist(image_path=f"{base_dir}/stardist/gex.tiff", 
             labels_npz_path=f"{base_dir}/stardist/gex.npz", 
             stardist_model="2D_versatile_fluo", 
             prob_thresh=0.05, 
             nms_thresh=0.5
            )

# 
b2c.insert_labels(adata, 
                  labels_npz_path=f"{base_dir}/stardist/gex.npz", 
                  basis="array", 
                  mpp=mpp, 
                  labels_key="labels_gex"
                 )


bdata = adata[mask]
#the labels obs are integers, 0 means unassigned
bdata = bdata[bdata.obs['labels_gex']>0]
bdata.obs['labels_gex'] = bdata.obs['labels_gex'].astype(str)
sc.pl.spatial(bdata, color=[None, "labels_gex"], img_key="0.5_mpp_150_buffer", basis="spatial_cropped_150_buffer")
plt.savefig(f"{base_dir}/stardist/array_expanded.png", bbox_inches='tight', dpi=300)
plt.close()


#the label viewer wants a crop of the processed image
#get the corresponding coordinates spanning the subset object
crop = b2c.get_crop(bdata, basis="array", mpp=mpp)
#GEX pops better with percentile normalisation performed
rendered = b2c.view_labels(image_path=f"{base_dir}/stardist/gex.tiff", 
                           labels_npz_path=f"{base_dir}/stardist/gex.npz", 
                           crop=crop,
                           stardist_normalize=True
                          )
plt.imshow(rendered)
plt.savefig(f"{base_dir}/stardist/array_rendered.png", bbox_inches='tight', dpi=300)
plt.close()

# 以 H&E 分割结果为主，如果某些区域 H&E 没认出细胞但 GEX 认出来了，就将这些“幸存”细胞补充进来，生成 labels_joint。
b2c.salvage_secondary_labels(adata, 
                             primary_label="labels_he_expanded", 
                             secondary_label="labels_gex", 
                             labels_key="labels_joint"
                            )


bdata = adata[mask]
#the labels obs are integers, 0 means unassigned
bdata = bdata[bdata.obs['labels_joint']>0]
bdata.obs['labels_joint'] = bdata.obs['labels_joint'].astype(str)
sc.pl.spatial(bdata, color=[None, "labels_joint_source", "labels_joint"], 
              img_key="0.5_mpp_150_buffer", basis="spatial_cropped_150_buffer")
plt.savefig(f"{base_dir}/stardist/array_label_expanded.png", bbox_inches='tight', dpi=300)
plt.close()


#%% bin2cell Step5. 将 Bin 映射到 Cell，生成单细胞级别的 AnnData对象
cdata = b2c.bin_to_cell(adata, labels_key="labels_joint", spatial_keys=["spatial", "spatial_cropped_150_buffer"])
cdata.write(f"{base_dir}/stardist/{sample_name}_stardist_adata.h5ad")


cell_mask = ((cdata.obs['array_row'] >= 1450) & 
             (cdata.obs['array_row'] <= 1550) & 
             (cdata.obs['array_col'] >= 250) & 
             (cdata.obs['array_col'] <= 450)
            )
ddata = cdata[cell_mask]
sc.pl.spatial(ddata, color=["bin_count", "labels_joint_source"], img_key="0.5_mpp_150_buffer", basis="spatial_cropped_150_buffer")
plt.savefig(f"{base_dir}/stardist/cell_mask.png", bbox_inches='tight', dpi=300)
plt.close()


crop = b2c.get_crop(ddata, basis="spatial", spatial_key="spatial_cropped_150_buffer", mpp=mpp)
#this is a string at this time
cdata.obs["labels_joint_source"] = cdata.obs["labels_joint_source"].astype("category")
img, legends = b2c.view_cell_labels(image_path=f"{base_dir}/stardist/he.tiff",
                                    labels_npz_path=f"{base_dir}/stardist/he.npz",
                                    cdata=cdata,
                                    crop=crop,
                                    fill_key="bin_count",
                                    border_key="labels_joint_source"
                                   )
plt.imshow(img)
plt.savefig(f"{base_dir}/stardist/spatial.png", bbox_inches='tight', dpi=300)
plt.close()







#%% 全图绘制
import matplotlib.pyplot as plt
import os

# 1. 确保 labels_joint_source 是分类变量，以便着色
cdata.obs["labels_joint_source"] = cdata.obs["labels_joint_source"].astype("category")

# 2. 调用 view_cell_labels，不传 crop 参数即代表全图
full_img, legends = b2c.view_cell_labels(
    image_path=f"{base_dir}/stardist/he.tiff", # 这里的 he.tiff 是你之前生成的 0.5mpp 图
    labels_npz_path=f"{base_dir}/stardist/he.npz",
    cdata=cdata,
    crop=None,                       # 关键：设为 None 则绘制全图
    fill_key="bin_count",            # 填充颜色反映每个细胞包含的 Bin 数量
    cont_cmap="magma",
    border_key="labels_joint_source", # 边界颜色反映细胞识别来源
    fill_label_weight=0.35,          # 适当降低透明度，更清晰地看到 H&E 背景
    thicken_border=False             # 全图视角下，边界不需要加粗，否则会糊成一团
)

# 3. 保存高清大图
plt.figure(figsize=(20, 20)) # 增加画布尺寸
plt.imshow(full_img)
plt.axis('off')
output_path = f"{base_dir}/stardist/Full_Tissue_Cell_Overlay.png"
plt.savefig(output_path, bbox_inches='tight', dpi=800)
print(f"全图已保存至: {output_path}")

# 4. 保存对应的全图图例
for key, fig in legends.items():
    fig.savefig(f"{base_dir}/stardist/full_legend_{key}.pdf", bbox_inches='tight')




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

cdata.write(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.h5ad")

sc.set_figure_params(dpi=50,fontsize=10,)
sc.pl.violin(cdata, 
             ['n_genes_by_counts', 
              'log1p_n_genes_by_counts', 
              'total_counts', 
              'log1p_total_counts', 
              'pct_counts_in_top_50_genes', 
              'pct_counts_in_top_100_genes',
              'bin_count'],
             jitter=0.05, 
             multi_panel=True)
plt.savefig(f"{base_dir}/stardist/violin.png", bbox_inches='tight', dpi=600)
plt.close()


sc.set_figure_params(dpi=100,fontsize=10,)
sc.tl.leiden(cdata,resolution=6,key_added='leiden')
sc.pl.umap(cdata,
           color=['leiden'],
           size=2,
           wspace=0.25,
           frameon=False)
plt.savefig(f"{base_dir}/stardist/umap.png", bbox_inches='tight', dpi=600)
plt.close()




# Confidence threshold
conf_th = 0

# Set figure parameters
sc.set_figure_params(dpi=100, dpi_save=2000, figsize=[10, 10], format='jpg')

# Plot for b2c he
sc.pl.spatial(cdata, color='leiden', 
              title='b2c he', size=20, img_key='hires', legend_fontsize=5,
              spot_size=1, frameon=False, )
plt.savefig(f"{base_dir}/stardist/spatial_leiden.png", bbox_inches='tight', dpi=600)
plt.close()



# 1. 确保 leiden 标签是分类变量（通常 sc.tl.leiden 后已经是了，但保险起见转换一下）
cdata.obs['leiden'] = cdata.obs['leiden'].astype('category')

print("正在生成基于 Leiden 聚类的全图细胞形状叠加图...")

# 2. 调用 view_cell_labels 进行像素级渲染
# fill_key 设置为 'leiden'，这样每个细胞的填充颜色就代表它的聚类群
img_leiden, legends_leiden = b2c.view_cell_labels(
    image_path=f"{base_dir}/stardist/he.tiff",
    labels_npz_path=f"{base_dir}/stardist/he.npz",
    cdata=cdata,
    crop=None,                 # 绘制全图
    fill_key="leiden",         # 关键：按聚类填充颜色
    cat_cmap="tab20",          # 使用适合多簇的调色盘，颜色更丰富
    fill_label_weight=0.6,     # 调高权重，使聚类颜色更鲜艳，容易区分
    border_key=None,           # 如果不需要边界线可以设为 None，或者设为 'leiden' 加深边缘
    thicken_border=False
)

# 3. 保存结果
plt.figure(figsize=(20, 20))
plt.imshow(img_leiden)
plt.axis('off')
output_leiden_path = f"{base_dir}/stardist/Full_Tissue_Leiden_Shape.png"
plt.savefig(output_leiden_path, bbox_inches='tight', dpi=800)

# 4. 保存图例
if 'leiden' in legends_leiden:
    legends_leiden['leiden'].savefig(f"{base_dir}/stardist/legend_leiden.pdf", bbox_inches='tight')



# 查看 CellTypist 的预测结果 (假设列名是 'predicted_labels')
img_celltype, legends_celltype = b2c.view_cell_labels(
    image_path=f"{base_dir}/stardist/he.tiff",
    labels_npz_path=f"{base_dir}/stardist/he.npz",
    cdata=cdata,
    crop=None,
    fill_key="predicted_labels", # 换成 CellTypist 的预测列名
    cat_cmap="tab20",
    fill_label_weight=0.5
)
plt.imsave(f"{base_dir}/stardist/Full_Tissue_CellTypist_Shape.png", img_celltype)

if 'predicted_labels' in legends_celltype:
    legends_celltype['predicted_labels'].savefig(f"{base_dir}/stardist/legend_predicted_labels.pdf", bbox_inches='tight')

cdata.write(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.h5ad")







#%% 单细胞数据集映射
adata_ref = sc.read_h5ad("/public3/Xinyu/3D_tissue/scRNA/nature_ref.h5ad")

# 确保 cdata 和 adata_ref 的基因名对齐 (使用 Symbol)
# Ingest 要求基因子集必须完全一致
common_genes = adata_ref.var_names.intersection(cdata.var_names)
adata_ref = adata_ref[:, common_genes].copy()
cdata_query = cdata[:, common_genes].copy()

# 参考集必须先运行过完整的降维流程
# 1. 重新进行高度标准化的降维流程
sc.pp.pca(adata_ref, n_comps=30)

# 2. 关键：强制重新计算 neighbors，增加 n_neighbors 数量
# 并且确保使用指定的 method='umap'，这有助于矩阵对齐
sc.pp.neighbors(adata_ref, n_neighbors=15, n_pcs=30, method='umap')

# 3. 运行 umap
sc.tl.umap(adata_ref)

# 运行映射
# obs=['CellSubType'] 表示你想从参考集迁移哪一列标签
# sc.tl.ingest(cdata_query, adata_ref, obs=['predicted.CellSubType'])
# # 尝试运行 ingest
# sc.tl.ingest(
#     cdata_query, 
#     adata_ref, 
#     obs=['predicted.CellSubType'], 
#     labeling_method='knn' # 显式指定使用 knn 标注
# )
# # 将映射后的标签写回原始的 cdata 对象
# cdata.obs['Flex_CellSubType'] = cdata_query.obs['CellSubType']
# cdata.obs['Flex_Score'] = cdata_query.obs['ingest_scores'] # 映射置信度
# 
# print("映射完成！细胞类型分布：")
# print(cdata.obs['Flex_CellSubType'].value_counts())


from sklearn.neighbors import KNeighborsClassifier

# 1. 获取参考集的 PCA 空间和标签
X_train = adata_ref.obsm['X_pca']
y_train = adata_ref.obs['predicted.CellSubType']

# 2. 将查询集 (Visium HD) 投影到参考集的 PCA 空间
# 计算公式：Query_PCA = Query_Expression @ Ref_PCA_Loadings
# 注意：这要求基因必须完全对齐（你之前已经做到了）
X_test = cdata_query.X @ adata_ref.varm['PCs']

# 3. 使用简单的 KNN 进行分类映射
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)

# 4. 预测标签并写回 cdata
cdata.obs['Flex_CellSubType'] = knn.predict(X_test)

print("手动映射完成！")

# 1. 将映射结果转换为分类变量 (Categorical)
cdata.obs['Flex_CellSubType'] = cdata.obs['Flex_CellSubType'].astype('category')

# 2. 现在重新运行绘图
img_celltype, legends_celltype = b2c.view_cell_labels(
    image_path=f"{base_dir}/stardist/he.tiff",
    labels_npz_path=f"{base_dir}/stardist/he.npz",
    cdata=cdata,
    crop=None,
    fill_key="Flex_CellSubType", 
    cat_cmap="tab20",           # 建议使用 tab20 以区分较多亚型
    fill_label_weight=0.5
)

# 3. 保存
plt.imsave(f"{base_dir}/stardist/Full_Tissue_Flex_CellSubType_Shape.png", img_celltype)

# 4. 保存图例
if 'Flex_CellSubType' in legends_celltype:
    legends_celltype['Flex_CellSubType'].savefig(f"{base_dir}/stardist/legend_Flex_CellSubType.pdf", bbox_inches='tight')

# 5. 最后保存更新后的 h5ad
# cdata.write(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.v3.h5ad")



#%% 单细胞数据集映射方式2
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# 1. 确保基因完全对齐 (这步你之前做过了，这里是安全检查)
common_genes = adata_ref.var_names.intersection(cdata.var_names)
adata_ref_sub = adata_ref[:, common_genes]
cdata_sub = cdata[:, common_genes]

# 2. 提取参考集的 PCA 载荷 (V) 和 基因均值 (μ)
# 只有减去参考集的均值，投影后的坐标才会在同一个空间
ref_pcs = adata_ref_sub.varm['PCs']
ref_mu = adata_ref_sub.var['mean'].values if 'mean' in adata_ref_sub.var else adata_ref_sub.X.mean(axis=0)

# 3. 计算 X_test_pca (核心公式：(X - μ) @ V)
# 如果 cdata_sub.X 是稀疏矩阵，需要先 toarray()
X_query_dense = cdata_sub.X.toarray() if hasattr(cdata_sub.X, "toarray") else cdata_sub.X
X_test_pca = (X_query_dense - ref_mu) @ ref_pcs

print(f"X_test_pca 计算完成，形状为: {X_test_pca.shape}")

# 1. 初始化 KNN 并拟合参考集
# n_neighbors=5 是常用值，如果数据量大可以尝试 15
knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1, weights='distance')
knn.fit(X_train, y_train)

# 2. 获取每个预测结果的概率矩阵 (n_samples, n_classes)
print("正在计算映射概率...")
probs = knn.predict_proba(X_test_pca)

# 3. 提取最高概率及其对应的标签
max_probs = probs.max(axis=1)
predicted_indices = probs.argmax(axis=1)
best_labels = knn.classes_[predicted_indices]

# 4. 设定过滤阈值 (针对 Tip cell 建议设在 0.6 - 0.7)
threshold = 0.6

# 5. 执行过滤逻辑：低于阈值的标记为 'Low_confidence'
final_labels = [
    label if prob >= threshold else "Low_confidence" 
    for label, prob in zip(best_labels, max_probs)
]

# 6. 写回原始 cdata 对象
cdata.obs['Flex_CellSubType_0.6'] = final_labels
cdata.obs['Flex_Score'] = max_probs

# 7. 统计结果
print(f"映射完成！在阈值 {threshold} 下：")
print(cdata.obs['Flex_CellSubType_0.6'].value_counts())

# 检查 Tip cell 的留存情况
n_tip = (cdata.obs['Flex_CellSubType_0.6'] == 'Tip cell').sum()
print(f"\n保留的高置信度 Tip cell 数量: {n_tip}")
cdata.write(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.v2.h5ad")





#%% 绘制单一细胞类型
# ================= 1. 准备数据 =================
# 创建只包含 Tip cell 的分类列，其余设为 NaN
cdata.obs['TipCell_Only'] = cdata.obs['Flex_CellSubType'].apply(
    lambda x: 'Tip cell' if x == 'Tip cell' else np.nan
).astype('category')

# ================= 2. 关键修复：手动注入颜色板 =================
# Scanpy/AnnData 存储分类颜色的标准位置是 uns['{column}_colors']
# 因为现在只有一个类别 'Tip cell'，所以列表里只需要一个颜色
cdata.uns['TipCell_Only_colors'] = ['#00FF00']  # 纯橙色

# ================= 3. 调用 bin2cell 绘制 =================
print(f"正在生成 Tip cell 的暗背景纯色叠加图...")

# 注意：这里 cat_cmap 设为 None，bin2cell 会自动去 uns 找对应的颜色
img_single, legends_single = b2c.view_cell_labels(
    image_path=f"{base_dir}/stardist/he.tiff",
    labels_npz_path=f"{base_dir}/stardist/he.npz",
    cdata=cdata,
    crop=None,
    fill_key="TipCell_Only", 
    
    # 修改点：不再传字典，传 None 让它自动读取我们刚刚设置的 uns 颜色
    cat_cmap=None, 
    
    # 亮度控制：0.2 左右会让背景显得非常暗
    fill_label_weight=0.8, 
    
    border_key=None,
    thicken_border=False
)

# ================= 4. 保存 =================
plt.figure(figsize=(20, 20))
plt.imshow(img_single)
plt.axis('off')
output_path = f"{base_dir}/stardist/Full_Tissue_TipCell_Orange_DarkHE_Fixed.png"
plt.savefig(output_path, bbox_inches='tight', dpi=800)
plt.close()
del cdata.obs['TipCell_Only']







import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据：创建一个只包含目标细胞类型的子集
# 这样做是为了在绘图时，图面上完全不出现其他细胞的点
target_celltype = 'Tip-like ECs'
adata_subset = cdata[cdata.obs['predicted_labels'] == target_celltype].copy()

# 1. 获取样本 ID (用于访问 uns 里的参数)
library_id = list(adata_subset.uns['spatial'].keys())[0]

# 2. 关键修复：强制设置缩放因子为 1.0
# 因为 'spatial_cropped_150_buffer' 已经是基于 0.5mpp 图像的像素坐标了
# 官方默认的 hires_scalef 会导致点被错误缩小
adata_subset.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef'] = 1.0

# 3. 创建画布
fig, ax = plt.subplots(figsize=(15, 15))

# 4. 绘图
sc.pl.spatial(
    adata_subset,
    color='predicted_labels',
    img_key="0.5_mpp_150_buffer",   # 确保使用 0.5mpp 的底图
    basis="spatial_cropped_150_buffer", # 关键：使用局部对齐坐标
    
    # --- 亮度与颜色控制 ---
    alpha_img=0.4,                  # 背景亮度：0.4 代表压暗背景，让点更亮
    palette={'Tip cell': '#00FF00'}, # 亮绿色，与暗色背景形成强对比
    spot_size=50,                   # 138个细胞较稀疏，可以适当调大点（30-60）
    
    show=False,
    frameon=False,
    ax=ax
)

# 5. 可选：让图像更深邃
# 即使 alpha_img 设了，背景还是白色画布。将背景设为黑色效果最佳：
ax.set_facecolor('black')

# 6. 保存
output_path = f"{base_dir}/stardist/TiplikeECs_Points_Perfect_Aligned.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='black')
print(f"完美对齐的点分布图已保存至: {output_path}")

#%% 卡阈值绘制
# ================= 1. 准备数据并过滤置信度 =================
target_celltype = 'Tip-like ECs'
confidence_threshold = 0.8  # 设置你的阈值，例如只保留置信度 > 0.6 的细胞

# 同时满足：类型匹配 且 置信度达标
mask = (cdata.obs['predicted_labels'] == target_celltype) & (cdata.obs['conf_score'] >= confidence_threshold)
adata_subset = cdata[mask].copy()

print(f"原始 {target_celltype} 数量: {(cdata.obs['predicted_labels'] == target_celltype).sum()}")
print(f"过滤后（Score > {confidence_threshold}）剩余数量: {adata_subset.n_obs}")

# ================= 2. 坐标校准 (保持不变) =================
library_id = list(adata_subset.uns['spatial'].keys())[0]
adata_subset.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef'] = 1.0

# ================= 3. 绘图 (注意颜色映射细节) =================
fig, ax = plt.subplots(figsize=(15, 15))

sc.pl.spatial(
    adata_subset,
    color='predicted_labels',
    img_key="0.5_mpp_150_buffer",
    basis="spatial_cropped_150_buffer",
    
    # 颜色设置：注意这里的 key 要和你 color 参数里的列名对应
    palette={target_celltype: '#00FF00'}, 
    alpha_img=0.4,
    spot_size=60, # 过滤后点变少了，可以适当再调大一点增加可视度
    
    show=False,
    frameon=False,
    ax=ax
)

ax.set_facecolor('black')

# 保存
output_path = f"{base_dir}/stardist/TiplikeECs_Filtered_Score{confidence_threshold}.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='black')
print(f"过滤后的点分布图已保存至: {output_path}")

#%% 多细胞类型绘制
# ================= 1. 定义想要展示的细胞类型和颜色 =================
# 建议：目标细胞用亮色，背景/对照细胞用暗色或中性色
target_types = ['Tip-like ECs', 'CMS2', 'CD4+ T cells', 'CD8+ T cells', 'Stalk-like ECs', 'SPP1+']

# 创建颜色字典
my_palette = {
    'Tip-like ECs': '#00FF00',   # 亮绿
    'Stalk-like ECs': '#0080FF', # 亮蓝
    'CMS2': '#F291B5',           # 红色
    'CD4+ T cells': '#F2A413',            # 黄色
    'CD8+ T cells': '#FFFFFF',   # 品红
    'SPP1+': '#FF00FF'          # 紫色
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
confidence_threshold = 0.5
mask = (cdata.obs['predicted_labels'].isin(target_types)) & (cdata.obs['conf_score'] >= confidence_threshold)
adata_multi = cdata[mask].copy()

# ================= 3. 绘图 =================
fig, ax = plt.subplots(figsize=(15, 15))

# 修复缩放因子
lib_id = list(adata_multi.uns['spatial'].keys())[0]
adata_multi.uns['spatial'][lib_id]['scalefactors']['tissue_hires_scalef'] = 1.0

sc.pl.spatial(
    adata_multi,
    color='predicted_labels',
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
output_multi = f"{base_dir}/stardist/Overlay_Tip_Tumor_Aligned.png"
plt.savefig(output_multi, dpi=600, bbox_inches='tight', facecolor='black')
print(f"多类型叠加图已保存至: {output_multi}")




# 1. 确保类别是 categorical
cdata.obs['predicted_labels'] = cdata.obs['predicted_labels'].astype('category')
n_cats = len(cdata.obs['predicted_labels'].cat.categories)

# 2. 方案：拼接 Scanpy 内置的多个 20 色板 (达到 40 色以上)
# Scanpy 默认带了多个离散色板，我们可以把它们连起来
from scanpy.plotting import palettes
all_colors = palettes.vega_20 + palettes.zeileis_28 + palettes.godsnot_102
my_palette = all_colors[:n_cats]

# 3. 将颜色存入 adata
cdata.uns['predicted_labels_colors'] = my_palette

# 4. 绘图
# 使用 shuffle=True 打乱绘图顺序，防止大群（如 CMS2）遮盖小群（如 Tip-like ECs）
sc.pl.umap(
    cdata, 
    color='predicted_labels', 
    size=1.0, 
    palette=my_palette,
    legend_loc='right margin', 
    frameon=False,
    title=f'UMAP: {n_cats} Categories (Merged Palettes)',
    show=False
)

# 5. 保存
output_path = f"{base_dir}/stardist/UMAP_CellTypist_MultiPalette_Full.pdf"
plt.savefig(output_path, dpi=1000, bbox_inches='tight')



#%% 绘制基因表达
import bin2cell as b2c
import matplotlib.pyplot as plt
import scanpy as sc


# 1. 设置基因名（确保在 cdata.var_names 中）
target_gene = "PECAM1" 

# 2. 检查并校准缩放因子 (针对 Visium HD 的必须步骤)
# 确保使用 bin2cell 对齐后的坐标系，并将缩放因子设为 1.0
library_id = list(cdata.uns['spatial'].keys())[0]
cdata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef'] = 1.0

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(12, 12))

sc.pl.spatial(
    cdata,
    color=target_gene,
    img_key="0.5_mpp_150_buffer",       # 这里的 key 需对应你 adata.uns['spatial'] 里的图像
    basis="spatial_cropped_150_buffer", # 关键：使用对齐后的局部像素坐标
    vmax = 0.5,
    # --- 样式控制 ---
    color_map="magma",                  # 推荐 'magma' 或 'viridis'，高亮表达
    alpha_img=0.3,                      # 调低背景亮度 (0.1-0.3)，突出基因表达
    spot_size=20,                       # 细胞很密，点要小一点；若看不清可调大至 40
    
    show=False,
    frameon=False,
    ax=ax
)

# 4. 可选：如果你完全不想要 H&E 背景 (纯黑背景模式)
# 只需要把 alpha_img 设为 0，并设置 axes 背景为黑色
# ax.set_facecolor('black') 
# sc.pl.spatial(..., alpha_img=0, ...)

# 5. 保存
output_path = f"{base_dir}/stardist/SC_Spatial_{target_gene}_Expression.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"基因 {target_gene} 的表达图已保存。")








import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# ================= 1. 核心设置与数据准备 =================
target_genes = ['ITGA5', 'ITGB1']  # 典型的 Tip cell 标志物，通常表达量很高但很局限

for target_gene in target_genes:
    print(f"\n正在处理基因: {target_gene} ...")
    # 检查基因是否存在
    if target_gene not in cdata.var_names:
        print(f"警告: 基因 {target_gene} 不在数据中。请检查拼写。")
        # 这里可以设置一个备选基因，例如 ESM1
        # target_gene = "ESM1" 

    # 获取该基因的表达量数据（转换为密集数组方便计算）
    gene_expression = cdata[:, target_gene].X.toarray().flatten()

    # ================= 2. 动态计算视觉增强阈值 (关键步骤) =================
    # 2.1 过滤掉表达量为 0 的细胞，只计算有表达的细胞的分位数
    positive_expression = gene_expression[gene_expression > 0]

    # 如果该基因几乎没有表达，设置默认值
    if len(positive_expression) == 0:
        print(f"警告: 基因 {target_gene} 在所有细胞中表达量均为 0。")
        v_min, v_max = 0, 1
    else:
        # 2.2 计算 99% 分位数作为上限 (vmax)
        # 这能确保颜色条的顶端卡在最亮的 1% 的细胞上，拉开颜色差异
        v_max = np.nanquantile(positive_expression, 0.99)

        # 2.3 设置低值过滤阈值 (vmin)
        # 过滤掉低于 0.5 的背景杂讯（假设数据已标准化）
        # 这样可以让低表达的 Stalk-like ECs 显示为深色，突出 Tip cell
        if target_gene == 'PECAM1':
            v_min = np.nanquantile(positive_expression, 0.5)
        else:
            v_min = 0.5

    # 确保 v_max 大于 v_min，防止绘图报错
    if v_max <= v_min:
        v_max = v_min + 1

    print(f"增强版视觉设置: vmin={v_min:.3f}, vmax={v_max:.3f}")

    # ================= 3. 绘图与视觉增强控制 =================
    # 3.1 创建一个全黑背景的画布，模拟荧光效果，让Magma色板更亮
    fig, ax = plt.subplots(figsize=(15, 15), facecolor='black')

    # 3.2 修复 Visium HD 的缩放因子 (针对 bin2cell 的必须步骤)
    library_id = list(cdata.uns['spatial'].keys())[0]
    cdata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef'] = 1.0

    # 3.3 绘图
    sc.pl.spatial(
        cdata,
        color=target_gene,
        img_key="0.5_mpp_150_buffer",       # 确保使用 0.5mpp 的底图
        basis="spatial_cropped_150_buffer", # 关键：使用局部对齐坐标

        # --- 核心视觉增强参数 ---
        vmin=v_min,                # 关键：过滤低值噪声，背景变暗
        vmax=v_max,                # 关键：拉高对比度，让高值细胞显得最亮
        color_map="magma",         # 推荐色板：在黑底下对比度极高（深蓝-红-黄-白）

        # --- 点的样式控制 ---
        spot_size=60,              # 加大点的大小（从20增加到60）。在23w细胞大图中，点必须大肉眼才能看到颜色。
        alpha=0.9,                 # 点略带透明，重叠区域（Tip cell聚集处）颜色会叠加得更深、更亮。

        # --- 背景控制 ---
        alpha_img=0.2,             # 进一步压暗 H&E 图像（从0.3降低到0.2），使其仅作为位置参考，不干扰基因表达。

        show=False,
        frameon=False,             # 去掉边框线
        ax=ax
    )

    # 3.4 优化 Colorbar (颜色条) 的可视度
    # 默认的 colorbar 字在黑底下看不见，我们手动调整
    colorbar_ax = plt.gcf().axes[-1]
    colorbar_ax.tick_params(labelsize=18, colors='white') # 调大字体，变为白色
    colorbar_ax.set_ylabel(f'{target_gene} Expression (LogNorm)', size=18, color='white')

    # 3.5 增加图表标题
    plt.title(f'Spatial Expression: {target_gene} (Vmax=p99)', size=25, color='white', pad=20)

    # ================= 4. 保存与检查 =================
    # 强制设置 axes 背景为黑色
    ax.set_facecolor('black')

    # 保存为高清 PNG
    # 必须设置 facecolor='black' 才能正确保存黑背景
    output_path_enhanced = f"{base_dir}/stardist/Spatial_{target_gene}_Expression_Enhanced.png"
    plt.savefig(output_path_enhanced, dpi=600, bbox_inches='tight', facecolor='black')
    plt.show()
    plt.close()
    print(f"增强版 {target_gene} 空间表达图（已过滤背景并加大点）已保存至: {output_path_enhanced}")









#%% 导出为Seurat数据格式

cdata = sc.read_h5ad(f"{base_dir}/stardist/{sample_name}_stardist_adata_annotated.h5ad")
# 1. 导出计数矩阵 (Matrix Market 格式)
# 注意：Seurat 习惯基因在行，细胞在列，所以这里需要转置 (.T)
mmwrite(f"{base_dir}/stardist/matrix.mtx", cdata.X.T)

# 2. 导出基因名和细胞名
pd.DataFrame(cdata.var_names).to_csv(f"{base_dir}/stardist/genes.tsv", sep='\t', index=False, header=False)
pd.DataFrame(cdata.obs_names).to_csv(f"{base_dir}/stardist/barcodes.tsv", sep='\t', index=False, header=False)

# 3. 导出元数据 (包含你之前算好的坐标、leiden等)
cdata.obs.to_csv(f"{base_dir}/stardist/metadata.csv")



