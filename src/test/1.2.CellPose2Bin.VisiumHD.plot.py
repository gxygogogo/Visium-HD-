import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.segmentation import find_boundaries

def plot_segmented_cells_expression(mask, adata, gene_name, x_range, y_range, 
                                     cmap='magma', scale_bar_um=50):
    """
    在指定 ROI 内，以细胞分割轮廓形式展示基因表达
    x_range, y_range: 像素坐标范围，例如 [10000, 12000]
    """
    # 1. 裁剪 Mask
    roi_mask = mask[y_range[0]:y_range[1], x_range[0]:x_range[1]].copy()
    
    # 2. 获取表达量数据 (确保数据已 log 标准化)
    if gene_name in adata.var_names:
        # 提取稀疏矩阵特定基因列并转为稠密数组
        exp_data = adata[:, gene_name].X.toarray().flatten()
    else:
        print(f"Gene {gene_name} not found in var_names.")
        return

    # 3. 建立 ID 到表达量的映射查找表
    # 假设 adata.obs['cells'] 存储的是 mask 中的原始像素 ID
    max_id = int(mask.max())
    lookup_table = np.zeros(max_id + 1, dtype=np.float32)
    
    # 获取 adata 中存在的 Cell IDs 及其在 obs 中的索引
    adata_cell_ids = adata.obs['cells'].values.astype(int)
    
    # 填充查找表：lookup_table[cell_id] = expression_value
    lookup_table[adata_cell_ids] = exp_data
    
    # 4. 映射：将 ROI Mask 中的 ID 替换为表达值
    # 这一步速度很快
    mapped_exp_image = lookup_table[roi_mask]
    
    # 5. 生成细胞边界轮廓线 (用于叠加，增强视觉效果)
    # mode='inner' 边界在细胞内，不吃背景
    boundaries = find_boundaries(roi_mask, mode='inner')
    
    # 6. 绘图
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # 绘制表达量基底
    # interpolation='nearest' 保持细胞边界清晰，不模糊
    im = ax.imshow(mapped_exp_image, cmap=cmap, interpolation='nearest', 
                   origin='upper') # Visium坐标原点通常在左上
    
    # 叠加白色半透明细胞边界 (可选，推荐)
    ax.imshow(boundaries, cmap=mcolors.ListedColormap(['none', 'white']), 
              alpha=0.3, origin='upper')
    
    # 美化
    plt.colorbar(im, ax=ax, label=f'{gene_name} log1p(Normalized Counts)', fraction=0.046, pad=0.04)
    ax.set_title(f'{gene_name} Expression in Segmented Cells (ROI)', fontsize=20)
    ax.axis('off')
    
    # 7. 添加比例尺 (Visium HD 1像素通常对应组织上的物理距离，需要通过scalefactors计算)
    # 假设 spaceranger scalefactors 中 microns_per_pixel 约为 0.18 (需检查你的json)
    # microns_per_pixel = 0.183 
    # bar_pixels = scale_bar_um / microns_per_pixel
    # ax.plot([50, 50 + bar_pixels], [roi_mask.shape[0]-50, roi_mask.shape[0]-50], color='white', lw=3)
    # ax.text(50 + bar_pixels/2, roi_mask.shape[0]-65, f'{scale_bar_um} µm', color='white', ha='center', fontsize=12)

    plt.show()
    fig.savefig(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{gene_name}_SEGMENTED_ROI.png', bbox_inches='tight', dpi=600)

# ================= 调用示例 =================
# 你需要根据 HE 图像找到感兴趣的区域像素坐标
# 比如你想看这片组织的正中心附近 2000x2000 像素的区域
h, w = full_mask.shape
roi_size = 3000
x_center, y_center = w // 2, h // 2

x_roi = [x_center - roi_size//2, x_center + roi_size//2]
y_roi = [y_center - roi_size//2, y_center + roi_size//2]

print(f"Plotting ROI: X={x_roi}, Y={y_roi}")

# 确保数据已标准化
if 'log1p' not in grouped_adata.uns:
    print("Pre-processing grouped_adata...")
    sc.pp.normalize_total(grouped_adata, target_sum=1e4)
    sc.pp.log1p(grouped_adata)

# 调用函数绘制 EPCAM
plot_segmented_cells_expression(full_mask, grouped_adata, 'EPCAM', x_roi, y_roi, cmap='magma')






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.segmentation import find_boundaries

def plot_segmented_gene(mask, adata, gene_name, x_range=None, y_range=None, cmap='magma'):
    """
    将基因表达量映射回分割掩码，并以细胞轮廓形式绘制
    :param mask: 你的 full_mask 数组
    :param adata: 聚合后的 grouped_adata
    :param gene_name: 想要绘制的基因名
    :param x_range: 像素坐标列表 [start, end]，若为 None 则绘制全图（慎用）
    :param y_range: 像素坐标列表 [start, end]
    """
    
    # 1. 提取基因表达向量
    if gene_name not in adata.var_names:
        print(f"基因 {gene_name} 不在数据中")
        return
    # 使用标准化后的数据
    exp_vector = adata[:, gene_name].X.toarray().flatten()
    
    # 2. 建立 ID 映射表
    # 假设 adata.obs['cells'] 存储的是 mask 中对应的像素值
    max_id = int(mask.max())
    lookup_table = np.zeros(max_id + 1, dtype=np.float32)
    
    cell_ids = adata.obs['cells'].values.astype(int)
    lookup_table[cell_ids] = exp_vector

    # 3. 裁剪区域 (ROI) 以保证渲染速度
    if x_range and y_range:
        sub_mask = mask[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    else:
        sub_mask = mask
        print("警告：正在尝试渲染全图，这可能需要消耗大量内存")

    # 4. 执行映射：将 Mask 中的 Cell ID 替换为表达值
    # 这是最核心的一步，将 ID 图片转为表达量图片
    exp_img = lookup_table[sub_mask]

    # 5. 计算细胞边界（用于勾勒白边，让细胞更清晰）
    boundaries = find_boundaries(sub_mask, mode='inner')

    # 6. 绘图
    plt.figure(figsize=(12, 12), dpi=600)
    
    # 绘制背景表达量
    im = plt.imshow(exp_img, cmap=cmap, interpolation='nearest')
    
    # 叠加半透明细胞边界线
    plt.imshow(boundaries, cmap=mcolors.ListedColormap(['none', 'white']), alpha=0.2)
    
    plt.colorbar(im, label=f'{gene_name} Normalized Expression', fraction=0.046, pad=0.04)
    plt.title(f'Cell Segmentation Expression: {gene_name}')
    plt.axis('off')
    plt.savefig(f'/public3/Xinyu/3D_tissue/Visium_HD/{sample}/{gene_name}_segmented_expression.png', bbox_inches='tight', dpi=600)

# --- 使用示例 ---

# 假设你想看图像中心的一个区域
h, w = full_mask.shape
center_x, center_y = w // 2, h // 2
offset = 1500  # 查看 3000x3000 像素的区域

# 确保数据已标准化
if 'log1p' not in grouped_adata.uns:
    sc.pp.normalize_total(grouped_adata, target_sum=1e4)
    sc.pp.log1p(grouped_adata)

# 绘制 EPCAM 基因在细胞分割后的原位表达
plot_segmented_gene(full_mask, grouped_adata, 'EPCAM', 
                    x_range=[20000, 30000], 
                    y_range=[20000, 30000])


