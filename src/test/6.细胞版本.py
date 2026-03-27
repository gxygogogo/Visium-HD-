

# library(Seurat)

# setwd('/public1/yuchen/3.DESI_project/um_20/11.ST_data/0.public_data/1.CRC.10X.visium.HD/Cell_segmentation/')

# obj <- readRDS('/public3/Xinyu/3D_tissue/Visium_HD/segmented_outputs/Segmented_merged.rds')
# write.table(obj@meta.data, '/public1/yuchen/3.DESI_project/um_20/11.ST_data/0.public_data/1.CRC.10X.visium.HD/Cell_segmentation/1.annotation.txt', sep='\t')







import json
import numpy as np
import cv2
import pandas as pd

df = pd.read_csv('/public1/yuchen/3.DESI_project/um_20/11.ST_data/0.public_data/1.CRC.10X.visium.HD/Cell_segmentation/1.annotation.txt', index_col=0, sep='\t')
df.index = (
    df.index
      .str.extract(r'cellid_(\d+)-')[0]
      .astype(int)
)
df = df[['orig.ident', 'predicted.id']]


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

COLOR_TUMOR = hex_to_bgr("#CBEEF3")
COLOR_TIP   = hex_to_bgr("#D62828")
COLOR_OTHER = hex_to_bgr("#f0f0f0")
df = df.copy()
df.index = df.index.astype(int)

# 加速查找
cell_type_dict = df["predicted.id"].to_dict()



# ========= 1. 读取 geojson =========
geojson_path = "/public3/Xinyu/3D_tissue/Visium_HD/segmented_outputs/cell_segmentations.geojson"

with open(geojson_path, "r") as f:
    data = json.load(f)

features = data["features"]

# ========= 4. 计算画布大小 =========
xs, ys = [], []

for feat in features:
    geom = feat["geometry"]
    if geom["type"] == "Polygon":
        for ring in geom["coordinates"]:
            ring = np.array(ring)
            xs.extend(ring[:, 0])
            ys.extend(ring[:, 1])
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            for ring in poly:
                ring = np.array(ring)
                xs.extend(ring[:, 0])
                ys.extend(ring[:, 1])

max_x = int(np.ceil(max(xs)))
max_y = int(np.ceil(max(ys)))

print(f"Canvas size: {max_y} x {max_x}")


# ========= 5. 创建白色背景 =========
canvas = np.ones((max_y + 1, max_x + 1, 3), dtype=np.uint8) * 255

# ========= 6. 画每个细胞 =========
for feat in features:
    cell_id = feat["properties"].get("cell_id", None)
    if cell_id is None:
        continue

    cell_type = cell_type_dict.get(cell_id, "Other")

    if cell_type == "Tumor":
        color = COLOR_TUMOR
    elif cell_type == "Tip cell":
        color = COLOR_TIP
    else:
        color = COLOR_OTHER

    geom = feat["geometry"]

    if geom["type"] == "Polygon":
        for ring in geom["coordinates"]:
            poly = np.array(ring, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [poly], color)

    elif geom["type"] == "MultiPolygon":
        for poly_coords in geom["coordinates"]:
            for ring in poly_coords:
                poly = np.array(ring, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(canvas, [poly], color)

# ========= 7. 保存 =========
out_path = (
    "/public1/yuchen/3.DESI_project/um_20/11.ST_data/"
    "0.public_data/1.CRC.10X.visium.HD/Cell_segmentation/"
    "cell_segmentation_colored.png"
)

cv2.imwrite(out_path, canvas)
print("Saved to:", out_path)