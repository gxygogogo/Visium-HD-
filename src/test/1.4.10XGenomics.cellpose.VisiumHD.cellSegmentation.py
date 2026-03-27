import os
import numpy as np
import pandas as pd
import scanpy as sc
import geopandas as gpd
import matplotlib.pyplot as plt

from tifffile import imread, imwrite
from shapely.geometry import Polygon, Point

from cellpose import models, core
from skimage.measure import regionprops, find_contours

#%% 分块细胞分割
# ================= 配置 =================

config = {

    "image_path":"/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_tissue_image.btf",
    "srdir":"/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/binned_outputs/square_002um",
    "output_dir":"/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/output_cellpose",

    # 分块数量
    "chunk_per_axis": 2
}

# ================= 主函数 =================

def run_cellpose_analysis(cfg):
    if not os.path.exists(cfg["output_dir"]):
        os.makedirs(cfg["output_dir"])

    print("===== Step1 读取图像 =====")
    img = imread(cfg["image_path"])

    if img.ndim == 3 and img.shape[0] < 10:
        img = np.transpose(img, (1,2,0))

    H, W = img.shape[:2]
    print("Image shape:", img.shape)

    # ================= Cellpose model =================
    print("===== Step2 加载 Cellpose =====")
    use_GPU = core.use_gpu()
    model = models.CellposeModel(
        gpu=use_GPU,
        pretrained_model="cyto"
    )

    # ================= 分块分割 =================
    chunk_per_axis = cfg["chunk_per_axis"]

    block_h = H // chunk_per_axis
    block_w = W // chunk_per_axis

    full_mask = np.zeros((H, W), dtype=np.uint32)
    constant = 1000000
    print("===== Step3 分块 Cellpose segmentation =====")
    block_id = 0
    for i in range(chunk_per_axis):
        for j in range(chunk_per_axis):
            r0 = i * block_h
            r1 = H if i == chunk_per_axis-1 else (i+1)*block_h
            c0 = j * block_w
            c1 = W if j == chunk_per_axis-1 else (j+1)*block_w
            print(f"Processing block {block_id+1}")
            tile = img[r0:r1, c0:c1]
            masks, flows, styles = model.eval(
                tile,
                channels=[0,0],
                diameter=None,
                flow_threshold=1,
                cellprob_threshold=0
            )
            masks = masks + constant * block_id
            full_mask[r0:r1, c0:c1] = masks
            block_id += 1
    full_mask = np.where(full_mask % constant == 0, 0, full_mask)
    mask_path = os.path.join(cfg["output_dir"], "cellpose_full_mask.tif")
    imwrite(mask_path, full_mask.astype(np.uint32))
    print("Mask saved:", mask_path)

    # ================= mask → polygon =================
    print("===== Step4 mask 转 polygon =====")
    regions = regionprops(full_mask)
    geometries = []

    for r in regions:
        cell_mask = full_mask == r.label
        contours = find_contours(cell_mask, 0.5)
        if len(contours) == 0:
            continue
        contour = max(contours, key=lambda x: x.shape[0])
        coords = [(p[1],p[0]) for p in contour]
        poly = Polygon(coords)

        if poly.is_valid and poly.area > 10:
            geometries.append(poly)
    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf["id"] = [f"ID_{i+1}" for i in range(len(gdf))]
    print("Detected cells:", len(gdf))

    # ================= 读取 Visium HD =================
    print("===== Step5 读取 Visium HD 数据 =====")

    raw_h5_file = os.path.join(cfg["srdir"], "filtered_feature_bc_matrix.h5")
    adata = sc.read_10x_h5(raw_h5_file)

    tissue_position_file = os.path.join(
        cfg["srdir"],
        "spatial/tissue_positions.parquet"
    )

    df_tissue_positions = pd.read_parquet(tissue_position_file)
    df_tissue_positions = df_tissue_positions.set_index("barcode")
    df_tissue_positions["index"] = df_tissue_positions.index

    adata.obs = pd.merge(
        adata.obs,
        df_tissue_positions,
        left_index=True,
        right_index=True
    )

    # ================= 创建空间点 =================
    print("===== Step6 构建空间点 =====")

    geometry_points = [
        Point(xy)
        for xy in zip(
            df_tissue_positions["pxl_col_in_fullres"],
            df_tissue_positions["pxl_row_in_fullres"]
        )
    ]

    gdf_coordinates = gpd.GeoDataFrame(
        df_tissue_positions,
        geometry=geometry_points
    )

    # ================= spatial join =================
    print("===== Step7 Spatial Join =====")
    result_spatial_join = gpd.sjoin(
        gdf_coordinates,
        gdf,
        how="left",
        predicate="within"
    )

    result_spatial_join["is_within_polygon"] = result_spatial_join["index_right"].isna()
    Result = result_spatial_join.loc[
        result_spatial_join["is_within_polygon"]
    ].copy()

    # ================= 保存 =================
    output_csv = os.path.join(
        cfg["output_dir"],
        "Cellpose_Nuclei_Barcode_Map.csv"
    )
    Result[["index","id","is_within_polygon"]].to_csv(
        output_csv,
        index=False
    )

    print("===== 完成 =====")
    print("Total nuclei:", len(gdf))
    print("Assigned bins:", len(Result))
    print("Result saved:", output_csv)


# ================= 运行 =================

run_cellpose_analysis(config)




#%% 全图细胞分割
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import scanpy as sc
import pickle
from tifffile import imread
from cellpose import models, core
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# ================= 配置区域 =================
config = {
    'image_path': '/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_tissue_image.btf',
    'srdir': '/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/binned_outputs/square_002um',
    'output_dir': '/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/output_cellpose_2',
    'model_type': 'cyto2',  # 可选 'cyto2' (细胞质+核) 或 'nuclei' (仅核)
    'diameter': 30,         # 细胞估计直径（像素），需根据你的分辨率调整
    'gpu': True             # 是否使用 GPU
}
# ============================================

def run_cellpose_analysis(cfg):
    if not os.path.exists(cfg['output_dir']):
        os.makedirs(cfg['output_dir'])

    # 1. 读取图像
    print("正在读取图像...")
    img = imread(cfg['image_path'])
    
    # 2. 初始化 Cellpose 模型
    print(f"正在加载 Cellpose 模型: {cfg['model_type']}...")
    use_gpu = core.use_gpu() if cfg['gpu'] else False
    model = models.Cellpose(gpu=use_gpu, model_type=cfg['model_type'])

    # 3. 运行分割
    # channels: [0, 0] 表示灰度图；如果是 H&E，通常是 [0, 0] 或特定通道
    print("开始 Cellpose 分割预测... 大图处理中...")
    masks, flows, styles, diams = model.eval(
        img, 
        diameter=cfg['diameter'], 
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0,
        resample=True
    )

    # 保存 Mask 结果（这是为了方便你后续核对对齐）
    with open(os.path.join(cfg['output_dir'], 'cellpose_masks.pckl'), 'wb') as f:
        pickle.dump(masks, f)

    # 4. 将 Mask 转换为多边形 (Polygons)
    print("正在提取细胞轮廓并转化为几何对象...")
    from skimage.measure import find_contours
    
    geometries = []
    cell_ids = []
    
    # 遍历每个被标记的细胞
    unique_masks = np.unique(masks)
    unique_masks = unique_masks[unique_masks > 0] # 剔除背景 0
    
    for m_id in unique_masks:
        # 寻找该 ID 的边界
        mask_roi = (masks == m_id).astype(np.uint8)
        contours = find_contours(mask_roi, 0.5)
        if len(contours) > 0:
            # 取最长的轮廓线作为细胞边界
            contour = max(contours, key=len)
            # 转换为 Shapely Polygon (注意：swap x and y)
            poly = Polygon([(c[1], c[0]) for c in contour])
            if poly.is_valid and poly.area > 5: # 剔除过小的碎片
                geometries.append(poly)
                cell_ids.append(f"Cell_{m_id}")

    gdf = gpd.GeoDataFrame({'id': cell_ids}, geometry=geometries)

    # 5. 加载数据并映射 (保持原逻辑不变)
    print("正在加载 Visium HD 数据并进行空间连接...")
    raw_h5_file = os.path.join(cfg['srdir'], "filtered_feature_bc_matrix.h5")
    adata = sc.read_10x_h5(raw_h5_file)
    
    tissue_pos = pd.read_parquet(os.path.join(cfg['srdir'], "spatial/tissue_positions.parquet"))
    tissue_pos = tissue_pos.set_index("barcode")
    
    # 构建坐标点
    geometry_points = [Point(xy) for xy in zip(tissue_pos["pxl_col_in_fullres"], tissue_pos["pxl_row_in_fullres"])]
    gdf_points = gpd.GeoDataFrame(tissue_pos, geometry=geometry_points)

    # 核心映射步骤
    result_join = gpd.sjoin(gdf_points, gdf, how="inner", predicate="within")
    
    # 6. 聚合生成表达矩阵 (这是之前代码缺少的部分)
    print("正在根据分割结果聚合基因表达量...")
    # 将 result_join 中的信息并回 adata.obs
    adata = adata[result_join.index].copy()
    adata.obs['cell_id'] = result_join['id']
    
    # 保存结果
    output_csv = os.path.join(cfg['output_dir'], "Cellpose_Barcode_Map.csv")
    adata.obs[['cell_id']].to_csv(output_csv)
    print(f"分析完成！映射表已保存至: {output_csv}")

# 执行
if __name__ == "__main__":
    run_cellpose_analysis(config)






#%% 分块改进版
import os
import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入进度条库
from tifffile import imread, imwrite
from shapely.geometry import Polygon, Point
from cellpose import models, core
from skimage.measure import regionprops, find_contours


# ================= 主函数 =================
def run_cellpose_analysis(cfg):
    if not os.path.exists(cfg["output_dir"]):
        os.makedirs(cfg["output_dir"])

    print("\n===== Step1 读取图像 =====")
    img = imread(cfg["image_path"])

    if img.ndim == 3 and img.shape[0] < 10:
        img = np.transpose(img, (1,2,0))

    H, W = img.shape[:2]
    print(f"图像尺寸: {img.shape}")

    print("\n===== Step2 加载 Cellpose =====")
    use_GPU = core.use_gpu()
    model = models.CellposeModel(gpu=use_GPU, pretrained_model="cyto")

    # ================= 分块分割 =================
    chunk_per_axis = cfg["chunk_per_axis"]
    block_h = H // chunk_per_axis
    block_w = W // chunk_per_axis
    full_mask = np.zeros((H, W), dtype=np.uint32)
    constant = 1000000

    print("\n===== Step3 分块 Cellpose 分割 =====")
    # 计算总块数用于进度条
    total_blocks = chunk_per_axis ** 2
    pbar_step3 = tqdm(total=total_blocks, desc="分割进度")

    block_id = 0
    for i in range(chunk_per_axis):
        for j in range(chunk_per_axis):
            r0 = i * block_h
            r1 = H if i == chunk_per_axis-1 else (i+1)*block_h
            c0 = j * block_w
            c1 = W if j == chunk_per_axis-1 else (j+1)*block_w
            
            tile = img[r0:r1, c0:c1]
            masks, flows, styles = model.eval(
                tile,
                channels=[0,0],
                diameter=None,
                flow_threshold=1,
                cellprob_threshold=0
            )
            # 只有非背景区域加偏移
            masks[masks > 0] = masks[masks > 0] + constant * block_id
            full_mask[r0:r1, c0:c1] = masks
            
            block_id += 1
            pbar_step3.update(1)
    pbar_step3.close()

    mask_path = os.path.join(cfg["output_dir"], "cellpose_full_mask.tif")
    imwrite(mask_path, full_mask)
    print(f"全图掩码已保存: {mask_path}")

    # ================= mask → polygon =================
    print("\n===== Step4 Mask 转 Polygon (几何化) =====")
    regions = regionprops(full_mask)
    geometries = []

    # 添加进度条，监控数万个细胞的转换
    for r in tqdm(regions, desc="多边形转化"):
        # 优化点：使用 r.image (局部掩码) 替代 full_mask == r.label (全局搜索)
        # 这能将此步速度提升数十倍
        local_mask = r.image.astype(np.uint8)
        # 补一圈零，防止轮廓在边界断开
        padded_mask = np.pad(local_mask, 1, mode='constant', constant_values=0)
        contours = find_contours(padded_mask, 0.5)
        
        if len(contours) == 0:
            continue
            
        contour = max(contours, key=lambda x: x.shape[0])
        # 转换回全局坐标：p[1]是本地x, p[0]是本地y, 需加上 bbox 的起始坐标偏移
        # 注意 padded_mask 补了一位，所以要减去 1
        min_y, min_x, _, _ = r.bbox
        coords = [(p[1] + min_x - 1, p[0] + min_y - 1) for p in contour]
        
        poly = Polygon(coords)
        if poly.is_valid and poly.area > 10:
            geometries.append(poly)

    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf["id"] = [f"ID_{i+1}" for i in range(len(gdf))]
    print(f"识别细胞数: {len(gdf)}")

    # ================= 读取 Visium HD =================
    print("\n===== Step5 读取 Visium HD 数据 =====")
    raw_h5_file = os.path.join(cfg["srdir"], "filtered_feature_bc_matrix.h5")
    adata = sc.read_10x_h5(raw_h5_file)

    tissue_position_file = os.path.join(cfg["srdir"], "spatial/tissue_positions.parquet")
    df_tissue_positions = pd.read_parquet(tissue_position_file)
    df_tissue_positions = df_tissue_positions.set_index("barcode")
    df_tissue_positions["index"] = df_tissue_positions.index

    # ================= 创建空间点 =================
    print("\n===== Step6 构建空间点 (Bin Coordinates) =====")
    # 使用 tqdm 包装 zip，监控 2um bins 的处理
    geometry_points = [
        Point(xy)
        for xy in tqdm(zip(
            df_tissue_positions["pxl_col_in_fullres"],
            df_tissue_positions["pxl_row_in_fullres"]
        ), total=len(df_tissue_positions), desc="构建点位")
    ]

    gdf_coordinates = gpd.GeoDataFrame(df_tissue_positions, geometry=geometry_points)

    # ================= spatial join =================
    print("\n===== Step7 Spatial Join (分配 Bin 到细胞) =====")
    # 注意：gpd.sjoin 是 C 层面的操作，无法直接加 tqdm，但它是最耗时的
    result_spatial_join = gpd.sjoin(
        gdf_coordinates,
        gdf,
        how="left",
        predicate="within"
    )

    result_spatial_join["is_within_polygon"] = ~result_spatial_join["index_right"].isna()
    Result = result_spatial_join.loc[result_spatial_join["is_within_polygon"]].copy()

    # ================= 保存 =================
    output_csv = os.path.join(cfg["output_dir"], "Cellpose_Nuclei_Barcode_Map.csv")
    Result[["index","id","is_within_polygon"]].to_csv(output_csv, index=False)

    # ================= 新增：保存 labels.pckl 和 polys.pckl =================
    print("\n===== Step8 保存 labels.pckl 和 polys.pckl =====")
    
    # 1. 保存 labels.pckl (即全图 mask)
    labels_path = os.path.join(cfg["output_dir"], "labels.pckl")
    with open(labels_path, 'wb') as f:
        pickle.dump(full_mask, f)
    print(f"labels.pckl 已保存: {labels_path}")

    # 2. 保存 polys.pckl
    # 为了兼容之前的 StarDist 格式，我们需要构造一个包含 'coord' 键的字典
    # 原始格式通常是 [array([y_coords]), array([x_coords])]
    stardist_style_polys = {'coord': []}
    for poly in geometries:
        x, y = poly.exterior.coords.xy
        # 注意：StarDist 习惯存为 [row, col] 即 [y, x]
        stardist_style_polys['coord'].append([np.array(y), np.array(x)])
    
    polys_path = os.path.join(cfg["output_dir"], "polys.pckl")
    with open(polys_path, 'wb') as f:
        pickle.dump(stardist_style_polys, f)
    print(f"polys.pckl 已保存: {polys_path}")

    print("\n===== 任务完成 =====")
    print(f"总细胞数: {len(gdf)}")
    print(f"成功关联的 Bins 数: {len(Result)}")
    print(f"映射结果已保存: {output_csv}")


# ================= 配置 =================
sample_list = ['Visium_HD_Human_Colon_Cancer_P5']
for sample in sample_list:
    print(f"\n\n===== 处理样本: {sample} =====")
    config = {
        "image_path":f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_tissue_image.btf",
        "srdir":f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/binned_outputs/square_002um",
        "output_dir":f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/output_cellpose",
        "chunk_per_axis": 2
    }

    run_cellpose_analysis(config)


