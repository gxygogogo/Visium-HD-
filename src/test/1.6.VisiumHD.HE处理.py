import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from skimage.segmentation import mark_boundaries

#%% 细胞轮廓叠加HE图像
# ================= 1. 配置路径 =================
sample = 'Visium_HD_Human_Colon_Cancer_P1'
img_path = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{sample}_tissue_image.btf"
mask_path = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/output_cellpose/labels.pckl"
output_path = f"/public3/Xinyu/3D_tissue/Visium_HD/{sample}/HE_Mask_Overlay.png"

# ================= 2. 加载数据 =================
print("正在加载图像和 Mask...")
img = imread(img_path)
# 确保 H&E 图像是 (H, W, 3) 格式
if img.ndim == 3 and img.shape[0] < 10:
    img = np.transpose(img, (1, 2, 0))

with open(mask_path, 'rb') as f:
    labels = pickle.load(f)

# ================= 3. 定义 ROI (局部叠加) =================
# 全图太大，建议先截取一个 3000x3000 的区域查看
r0, r1, c0, c1 = 19000, 23000, 15000, 19000 
img_roi = img[r0:r1, c0:c1]
labels_roi = labels[r0:r1, c0:c1]

# ================= 4. 叠加绘图 =================
print("正在生成叠加图...")
plt.figure(figsize=(15, 15), dpi=600)

# 方法 A: 绘制彩色边界线 (适合观察细胞核/边缘对齐)
# color=(1, 0, 0) 表示红色边界
overlay_img = mark_boundaries(img_roi, labels_roi, color=(1, 1, 0), mode='inner')

plt.imshow(overlay_img)
plt.title(f"H&E Overlay with Cell Boundaries - {sample}")
plt.axis('off')

# 保存结果
plt.savefig(output_path, bbox_inches='tight', dpi=600)
plt.show()



#%% 细胞轮廓+基因表达+HE图像叠加
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.segmentation import find_boundaries
from tifffile import imread
import pickle
import scanpy as sc

cell_adata = sc.read_h5ad('/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Visium_HD_Human_Colon_Cancer_P1_cellpose_adata.h5ad')

# ================= 1. 数据准备 =================
# 假设你已经有了 cell_adata, labels (mask), 和原始 img
gene_name = 'PIEZO1' 

# 提取表达量并建立映射表 (Lookup Table)
exp_vector = cell_adata[:, gene_name].X.toarray().flatten()
lut = np.zeros(labels.max() + 1, dtype=np.float32)
cell_indices = [int(str(name).split('_')[-1]) for name in cell_adata.obs_names]
lut[cell_indices] = exp_vector

# 生成像素级表达图，并将背景 (ID=0) 设为透明 (NaN)
gene_img = lut[labels].astype(float)
gene_img[labels == 0] = np.nan 

# 计算轮廓
boundaries = find_boundaries(labels, mode='inner')

# ================= 2. 局部 ROI 截取 =================
# 选一个感兴趣的区域
r0, r1, c0, c1 = 19000, 23000, 15000, 19000
img_roi = img[r0:r1, c0:c1]
gene_roi = gene_img[r0:r1, c0:c1]
boundary_roi = boundaries[r0:r1, c0:c1]

# ================= 3. 多层叠加绘图 =================
plt.figure(figsize=(15, 15), dpi=300)

# --- 第一层：H&E 底图 ---
# 稍微降低亮度 (alpha)，让上面的基因表达颜色更突出
plt.imshow(img_roi, alpha=0.8)

# --- 第二层：基因表达填充 ---
# 使用带有透明度的 Colormap
v_max = np.percentile(exp_vector[exp_vector > 0], 98) if np.any(exp_vector > 0) else 1.0
current_cmap = plt.cm.magma.copy()
current_cmap.set_bad(alpha=0) # 让背景完全透明，露出 HE

im = plt.imshow(gene_roi, 
                cmap=current_cmap, 
                vmin=0, 
                vmax=v_max, 
                alpha=0.6, # 调整透明度，0.6 可以同时看到细胞内的表达颜色和底下的 HE 纹理
                interpolation='nearest')

# --- 第三层：细胞轮廓 ---
# 使用白色轮廓，增加空间结构的清晰度
contour_cmap = mcolors.ListedColormap(['none', 'white'])
plt.imshow(boundary_roi, cmap=contour_cmap, alpha=0.3)

# ================= 4. 装饰与保存 =================
plt.title(f'H&E + {gene_name} Expression + Cell Contours', fontsize=18, pad=20)
cb = plt.colorbar(im, fraction=0.046, pad=0.04)
cb.set_label(f'{gene_name} LogExp', fontsize=12)

plt.axis('off')
plt.savefig(f'/public3/Xinyu/3D_tissue/Visium_HD/Visium_HD_Human_Colon_Cancer_P1/Cellpose_HE_Expression_Overlay_{gene_name}.png', bbox_inches='tight', dpi=600)
plt.show()

