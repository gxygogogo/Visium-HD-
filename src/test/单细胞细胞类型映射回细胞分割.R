library(Seurat)
nature_data <- readRDS('/public1/yuchen/3.DESI_project/um_20/4.scRNA-seq/2.CellType_Endothelial/run.datasets.onebyone/2.CancerCell_Deng_GSE205506_analisis/3.refine_umap/7.big_umap/All_cells.rds')
nature_data <- readRDS('/public1/yuchen/3.DESI_project/um_20/4.scRNA-seq/2.CellType_Endothelial/1.merge_3_data/merged.v7.rds')
t_merged.counts = Read10X('/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/stardist/feature')
t_merged <- CreateSeuratObject(counts = t_merged.counts)
t_merged <- NormalizeData(t_merged)
t_merged <- FindVariableFeatures(t_merged)
t_merged <- ScaleData(t_merged)
t_merged <- RunPCA(t_merged, npcs = 30)
DefaultAssay(nature_data) <- "RNA"


##########################################################################################  t_merged  #########################################################################################
t_merged.anchors <- FindTransferAnchors(reference = nature_data, query = t_merged, dims = 1:30)
# 将查询数据集映射到参考数据集上
t_merged_predictions <- TransferData(anchorset = t_merged.anchors, refdata = nature_data$CellSubType, dims = 1:30)
# 添加预测出的信息
t_merged <- AddMetaData(t_merged, metadata = t_merged_predictions)

saveRDS(t_merged, '/public3/Xinyu/3D_tissue/Visium_HD/segmented_outputs/Segmented_merged.rds')

t_merged <- readRDS('/public3/Xinyu/3D_tissue/Visium_HD/segmented_outputs/Segmented_merged.rds')
# 确保设置了正确的 Assay
DefaultAssay(t_merged) <- "RNA"

# 绘制离散的细胞类型分布
pdf('/public3/Xinyu/3D_tissue/Visium_HD/Predicted_CellTypes.pdf', width = 8, height = 6)
# 注意：TransferData 生成的得分通常存放在名为 "prediction.score" 的 Assay 中
# 如果你在 meta.data 里能看到这些列，可以直接绘制
SpatialFeaturePlot(t_merged, features = "prediction.score.CapECs_1", pt.size.factor = 1.6)
dev.off()


t_merged <- Load10X_Spatial(data.dir = "/public3/Xinyu/3D_tissue/Visium_HD/segmented_outputs/", filename = "filtered_feature_cell_matrix.h5")





# 安装（如果未安装）
# if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
# remotes::install_github("mojaveazure/seurat-disk")

library(Seurat)
library(SeuratDisk)

# 加载数据
nature_data <- readRDS('/public1/yuchen/3.DESI_project/um_20/4.scRNA-seq/2.CellType_Endothelial/1.merge_3_data/merged.v7.rds')

# 确保 DefaultAssay 是你想导出的那个（通常是 RNA）
DefaultAssay(nature_data) <- "RNA"

# 保存为中间格式
SaveH5Seurat(nature_data, filename = "/public3/Xinyu/3D_tissue/scRNA/nature_ref.h5Seurat", overwrite = TRUE)

# 转换为 h5ad (Scanpy 格式)
Convert("/public3/Xinyu/3D_tissue/scRNA/nature_ref.h5Seurat", dest = "h5ad", overwrite = TRUE)
