"""
下面是将整体点云划分为30*30的block的情况
可以合并点云数较少的block，+表示点云数较少的block索引
训练集
(13+, 26, 39+); (20+, 21); (33+, 34); (42+, 54+, 55); (43+, 56); (44+, 57); (45+, 58); (46+, 59);
(77, 78+); (79+, 92); (90, 91+); (114+, 115, 116+); (105, 118+); (145+, 158+, 171+); (150, 151+);
(163, 164+); (159, 172+); (160, 173+); (161, 174+);
验证集
(4+, 10); (5+, 11, 12+); (13+, 14); (17, 18+); (19+, 20); (25+, 26); (23, 29+); (40, 46+);
(50, 51+); (67, 68+); (77+, 78, 79, 80+); (97+, 98); (100+, 101, 102+);
"""

import os
import utils
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from mpl_toolkits.mplot3d import Axes3D


format_flag = True
print("保留原始格式进行存储：", format_flag)

# train_or_eval = "train"
# merge_blocks_list = [
#     (13, 26, 39), (20, 21), (33, 34), (42, 54, 55), (43, 56), (44, 57), (45, 58),
#     (46, 59), (77, 78), (79, 92), (90, 91), (114, 115, 116), (105, 118), (145, 158, 171),
#     (150, 151), (163, 164), (159, 172), (160, 173), (161, 174)
# ]

train_or_eval = "eval"
merge_blocks_list = [
    (4, 10), (5, 11, 12), (13, 14), (17, 18), (19, 20), (25, 26), (23, 29), (40, 46),
    (50, 51), (67, 68,), (77, 78, 79, 80), (97, 98), (100, 101, 102)
]

if format_flag:
    isprs_processed_path = "/dataset/Vaihingen3D/processed_no_rgb_raw_format"
else:
    isprs_processed_path = "/dataset/Vaihingen3D/processed_no_rgb"

raw_blocks_path = os.path.join(isprs_processed_path, train_or_eval)
merge_blocks_path = os.path.join(isprs_processed_path, train_or_eval + "_merge")
if os.path.exists(merge_blocks_path):
    shutil.rmtree(merge_blocks_path)
# 将原始的block复制一份
shutil.copytree(raw_blocks_path, merge_blocks_path)

raw_blocks_list = os.listdir(raw_blocks_path)
for merge_block in merge_blocks_list:
    merge_data = np.empty(shape=(0, 7))     # 建立一个空的np数组
    merge_block_path = os.path.join(merge_blocks_path, "block")
    for block in merge_block:
        raw_block_path = os.path.join(raw_blocks_path, "block_{}.txt".format(block))
        block_data = np.loadtxt(raw_block_path)
        merge_data = np.concatenate((merge_data, block_data), axis=0)
        merge_block_path = merge_block_path + "_{}".format(block)
        # 删除合并之前的block数据
        os.remove(os.path.join(merge_blocks_path, "block_{}.txt".format(block)))
    merge_block_path = merge_block_path + ".txt"

    if format_flag:
        # 保留原始格式的方式进行存储
        f = open(merge_block_path, "w")
        for i in range(merge_data.shape[0]):
            line_content = "{:.2f} {:.2f} {:.2f} {:d} {:d} {:d} {:d}\n".format(
                merge_data[i][0], merge_data[i][1], merge_data[i][2],
                int(merge_data[i][3]), int(merge_data[i][4]),
                int(merge_data[i][5]), int(merge_data[i][6])
            )
            f.write(line_content)
        f.close()
    else:
        # 使用numpy方式进行存储，数据类型会变为np.float64
        np.savetxt(merge_block_path, merge_data)

print("Merge done!")


"""画图初始化"""
# 三维作图初始化
fig3d = plt.figure(1)
ax3d = Axes3D(fig3d)
ax3d.view_init(elev=50, azim=50)
ax3d.axis("off")
# 二维作图初始化
fig2d, ax2d = plt.subplots()
ax2d.axis("off")

num_points = 0
all_merge_blocks = os.listdir(merge_blocks_path)
for merge_block in all_merge_blocks:
    block_data = np.loadtxt(os.path.join(merge_blocks_path, merge_block))[:, 0:3]
    x_min, x_max, y_min, y_max, z_min, z_max = utils.find_min_max(block_data)
    num_points += block_data.shape[0]

    # 画出三维散点图
    ax3d.scatter(block_data[:, 0], block_data[:, 1], block_data[:, 2], s=1, alpha=0.2)
    # 画出三维划分block的包裹框
    utils.plot_3d_linear_rect(ax3d, x_min, y_min, x_max, y_max, block_data, default_z=(z_min, z_max))

    # 画出二维投影散点图
    ax2d.scatter(block_data[:, 0], block_data[:, 1], s=1, alpha=0.2)
    # 画出二维投影包裹框
    rect = mpathes.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color="r", fill=False)
    ax2d.add_patch(rect)

print("Num points: ", num_points)
fig3d.savefig(os.path.join(isprs_processed_path, train_or_eval + "_merge3d.png"))
fig2d.savefig(os.path.join(isprs_processed_path, train_or_eval + "_merge2d.png"))
print("Plot done!")
