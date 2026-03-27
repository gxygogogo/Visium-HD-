library(Seurat)
library(Matrix)
library(future) # 开启并行，否则 28万细胞会非常慢

# 开启并行计算 (使用 8-16 个核心)
plan("multisession", workers = 8)
options(future.globals.maxSize = 80 * 1024^3) # 允许使用 80GB 内存

# 1. 加载数据
data_path <- "/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/SeuratInput/"
counts <- readMM(paste0(data_path, "counts.mtx"))
meta <- read.csv(paste0(data_path, "metadata.csv"), row.names = 1)
genes <- read.csv(paste0(data_path, "genes.csv"))

# 2. 处理重复基因名并创建对象
rownames(counts) <- make.unique(as.character(genes$X)) # 假设第一列是基因名
colnames(counts) <- rownames(meta)
t_merged <- CreateSeuratObject(counts = counts)
t_merged <- NormalizeData(t_merged)
t_merged <- FindVariableFeatures(t_merged)
t_merged <- ScaleData(t_merged)
t_merged <- RunPCA(t_merged, npcs = 30)

# 4. 加载参考数据 (nature_data)
nature_data <- readRDS('/public1/yuchen/3.DESI_project/um_20/4.scRNA-seq/2.CellType_Endothelial/1.merge_3_data/merged.v7.rds')
DefaultAssay(nature_data) <- "RNA"

# 5. 执行 Label Transfer
# 寻找锚点
anchors <- FindTransferAnchors(reference = nature_data, query = t_merged, dims = 1:30)

# 传递标签 (CellSubType 为参考数据集中的细胞类型列)
predictions <- TransferData(anchorset = anchors, refdata = nature_data$CellSubType, dims = 1:30)
t_merged <- AddMetaData(t_merged, metadata = predictions)

# 6. 注入空间坐标以便画图
spatial_coords <- read.csv(paste0(data_path, "spatial_coords.csv"), row.names = 1)
t_merged[["spatial"]] <- CreateDimReducObject(
  embeddings = as.matrix(spatial_coords[colnames(t_merged), ]), 
  key = "coord_", 
  assay = DefaultAssay(t_merged)
)

saveRDS(t_merged, "/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Annotated_t_merged.rds")


