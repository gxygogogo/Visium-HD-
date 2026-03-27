import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import geopandas as gpd
import scanpy as sc
import polars as pl
import pickle
from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from shapely.geometry import Polygon, Point
from scipy import sparse
from matplotlib.colors import ListedColormap
import pyarrow
from csbdeep.io import save_tiff_imagej_compatible
from stardist import export_imagej_rois

# ================= 配置区域：请在此处修改你的路径 =================
sample = 'Visium_HD_Human_Colon_Cancer_P5'
config = {
    'image_path': f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_tissue_image.btf', # 图像路径
    'srdir': f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/binned_outputs/square_002um', # SpaceRanger输出目录
    'output_dir': f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/output_stardist', # 结果保存目录
    
    # 如果你想处理全图，请将 rmax 和 cmax 设为 None
    # 如果只想处理一个小区域进行测试，请设置像素范围
    'rmin': 0, 
    'rmax': None, 
    'cmin': 0, 
    'cmax': None 
}
# ================================================================

def run_stardist_analysis(cfg):
    # 创建输出目录
    if not os.path.exists(cfg['output_dir']):
        os.makedirs(cfg['output_dir'])
        print(f"创建输出目录: {cfg['output_dir']}")

    # 1. 读取图像
    print("正在读取图像...")
    img = imread(cfg['image_path'])
    print(f"原始图像形状: {img.shape}")

    # 裁剪图像（如果指定了范围）
    if cfg['rmax'] is not None:
        img = img[cfg['rmin'] : cfg['rmax'], cfg['cmin'] : cfg['cmax']]
        save_tiff_imagej_compatible(os.path.join(cfg['output_dir'], 'img_Stardist_cropped.tif'), img, axes='YXC')
        print(f"已保存裁剪图像，形状: {img.shape}")

    # 2. 加载 StarDist 预训练模型
    # '2D_versatile_he' 适用于 H&E 染色
    # 如果是 DAPI 荧光染色，可以尝试 '2D_versatile_fluo'
    print("正在加载 StarDist 模型...")
    model = StarDist2D.from_pretrained('2D_versatile_he')

    # 3. 图像归一化
    print("正在归一化图像...")
    img_norm = normalize(img, 5, 95)

    # 4. 运行预测
    # predict_instances_big 适合处理 Visium HD 的大图
    print("正在进行细胞分割预测 (predict_instances_big)... 这可能需要较长时间...")
    labels, polys = model.predict_instances_big(
        img_norm, 
        axes='YXC', 
        block_size=4096, 
        prob_thresh=0.01, 
        nms_thresh=0.001, 
        min_overlap=128, 
        context=128, 
        normalizer=None, 
        n_tiles=(4,4,1)
    )

    # 保存中间结果
    with open(os.path.join(cfg['output_dir'], 'labels.pckl'), 'wb') as f:
        pickle.dump(labels, f)
    with open(os.path.join(cfg['output_dir'], 'polys.pckl'), 'wb') as f:
        pickle.dump(polys, f)

    # 5. 转化为 GeoDataFrame
    print("正在转化多边形几何数据...")
    geometries = []
    for nuclei in range(len(polys['coord'])):
        # 注意：StarDist 返回的是 (row, col)，Shapely 需要 (x, y) 即 (col, row)
        coords = [(y, x) for x, y in zip(polys['coord'][nuclei][0], polys['coord'][nuclei][1])]
        geometries.append(Polygon(coords))

    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf["id"] = [f"ID_{i+1}" for i in range(len(gdf))]

    # 6. 加载 Visium HD 数据 (2um bin)
    print("正在加载 Visium HD 基因数据...")
    raw_h5_file = os.path.join(cfg['srdir'], "filtered_feature_bc_matrix.h5")
    adata = sc.read_10x_h5(raw_h5_file)

    tissue_position_file = os.path.join(cfg['srdir'], "spatial/tissue_positions.parquet")
    df_tissue_positions = pd.read_parquet(tissue_position_file)
    df_tissue_positions = df_tissue_positions.set_index("barcode")
    df_tissue_positions["index"] = df_tissue_positions.index

    # 合并坐标到 adata
    adata.obs = pd.merge(adata.obs, df_tissue_positions, left_index=True, right_index=True)

    # 7. 空间映射 (Point in Polygon)
    print("正在执行空间连接 (Spatial Join)... 将 2um bins 分配给细胞核...")
    geometry_points = [
        Point(xy) for xy in zip(
            df_tissue_positions["pxl_col_in_fullres"],
            df_tissue_positions["pxl_row_in_fullres"]
        )
    ]
    gdf_coordinates = gpd.GeoDataFrame(df_tissue_positions, geometry=geometry_points)

    # 执行 Spatial Join
    result_spatial_join = gpd.sjoin(gdf_coordinates, gdf, how="left", predicate="within")
    
    # 标记哪些 bin 在细胞核内
    result_spatial_join["is_within_polygon"] = ~result_spatial_join["index_right"].isna()
    
    # 筛选结果
    Result = result_spatial_join.loc[result_spatial_join['is_within_polygon']].copy()
    
    # 保存映射表
    output_csv = os.path.join(cfg['output_dir'], "Nuclei_Barcode_Map.csv")
    Result[['index', 'id', 'is_within_polygon']].to_csv(output_csv, index=False)
    
    print(f"分析完成！映射表已保存至: {output_csv}")
    print(f"识别到的细胞核总数: {len(gdf)}")
    print(f"成功分配到细胞核的 bins 总数: {len(Result)}")

run_stardist_analysis(config)

