import os
import utils
import shutil
import numpy as np


train_or_eval = "train"
# train_or_eval = "eval"

"""建立存储gt_color的路径"""
gt_color_root = "/dataset/Vaihingen3D/gt_color"
gt_color_blocks_path = os.path.join(gt_color_root, train_or_eval)
if os.path.exists(gt_color_blocks_path):
    shutil.rmtree(gt_color_blocks_path)
os.makedirs(gt_color_blocks_path)

class2rgb = {
    0: [255, 105, 180],      # Powerline: hot pink
    1: [170, 255, 127],      # Low vegetation: 黄绿色
    2: [128, 128, 128],      # Impervious surfaces: gray
    3: [255, 215, 0],        # Car: gold
    4: [0, 191, 255],        # Fence: deep sky blue
    5: [0, 0, 127],          # Roof: blue
    6: [205, 133, 0],        # Facade: orange
    7: [160, 32, 240],       # Shrub: purple
    8: [9, 120, 26],         # Tree: green
}

isprs_processed_root = "/dataset/Vaihingen3D/processed_no_rgb"
merge_blocks_path = os.path.join(isprs_processed_root, train_or_eval + "_merge")

all_merge_blocks = os.listdir(merge_blocks_path)
for blocks in all_merge_blocks:
    blocks_path = os.path.join(merge_blocks_path, blocks)
    blocks_data = np.loadtxt(blocks_path)
    labels = blocks_data[:, 6]
    gt_color = np.zeros((blocks_data.shape[0], 3))

    # 将点云的类别转换为颜色
    for i, ele in enumerate(labels):
        gt_color[i] = class2rgb[ele]
    gt_color = gt_color.astype(np.uint8)

    """将整个区域的点云写入文件中"""
    utils.write_ply(
        os.path.join(gt_color_blocks_path, blocks[:-4] + ".ply"),
        [blocks_data[:, 0:3], gt_color],
        ["x", "y", "z", "gt_r", "gt_g", "gt_b"]
    )

print("Done!")
