"""
在训练集划分block
"""

import os
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from mpl_toolkits.mplot3d import Axes3D


format_flag = False
print("保留原始格式进行存储：", format_flag)
isprs_raw_root = "/dataset/Vaihingen3D"
isprs_raw_train = os.path.join(isprs_raw_root, "Vaihingen3D_Traininig.pts")

# 读取原始数据的最小坐标
raw_train_data = np.loadtxt(isprs_raw_train)
raw_x_min = np.min(raw_train_data[:, 0])
raw_y_min = np.min(raw_train_data[:, 1])

count_block = 0     # 划分block的计数器

"""划分block的超参数设置"""
stride_x, stride_y = 30.0, 30.0     # 划分block的步进幅度
block_x, block_y = 30.0, 30.0       # 每个block的大小
threshold = 2000                    # 记录少于threshold个点云的block
less_threshold_list = []

"""画图初始化"""
# 三维作图初始化
fig3d = plt.figure(1)
ax3d = Axes3D(fig3d)
ax3d.view_init(elev=0, azim=20)
# 二维作图初始化
fig2d, ax2d = plt.subplots()
ax2d.axis("off")

# 判断processed_path目录是否存在，不存在则建立该目录
if format_flag:
    processed_path = os.path.join(isprs_raw_root, "processed_no_rgb_raw_format")
else:
    processed_path = os.path.join(isprs_raw_root, "processed_no_rgb")
if not os.path.exists(processed_path):
    os.makedirs(processed_path)
# 判断record_num_points_train文件是否存在，存在则删除该文件
record_num_points_file = os.path.join(processed_path, "record_num_points_train.txt")
if os.path.exists(record_num_points_file):
    os.remove(record_num_points_file)
# 判断record_coordinates_train文件是否存在，存在则删除该文件
record_coordinates_file = os.path.join(processed_path, "record_coordinates_train.txt")
if os.path.exists(record_coordinates_file):
    os.remove(record_coordinates_file)
# 判断other_file文件是否存在，存在则删除该文件
other_file = os.path.join(processed_path, "other_train.txt")
if os.path.exists(other_file):
    os.remove(other_file)
# 判断train_blocks_path是否存在，不存在则建立该目录
train_blocks_path = os.path.join(processed_path, "train")
if not os.path.exists(train_blocks_path):
    os.makedirs(train_blocks_path)


num_block_points_list = []
other_f = open(other_file, "w")

"""将train区域进行划分"""
train_data = np.loadtxt(isprs_raw_train)
x_min, x_max, y_min, y_max, z_min, z_max = utils.find_min_max(train_data[:, 0:3])

start_x, start_y = x_min, y_min     # 划分block开始的位置
reset_x = x_min                     # 循环结束后，重置起始位置

non_zero_block = 0      # 含有点云的block数目
flag_x = False      # 内循环是否已经结束标致位
flag_y = False      # 外循环是否已经结束标致位

while start_y <= y_max:
    end_y = start_y + block_y
    # 外循环中被划分的block超出边界，则划分的时候让边界为最大值，保证block的边界和原始点云边界相同
    if end_y > y_max:
        # start_y = y_max - block_y     # 是否将start_y进行回退
        end_y = y_max + 1           # 将边界的点云也包括
        flag_y = True

    while start_x <= x_max:
        end_x = start_x + block_x
        # 内循环中被划分的block超出边界，则划分的时候让边界为最大值，保证block的边界和原始点云边界相同
        if end_x > x_max:
            # start_x = x_max - block_x     # 是否将start_x进行回退
            end_x = x_max + 1       # 将边界的点云也包括
            flag_x = True

        # 找到在划定的XY区域中的所有点云
        points_index = np.where((train_data[:, 0] >= start_x) & (train_data[:, 0] < end_x) &
                                (train_data[:, 1] >= start_y) & (train_data[:, 1] < end_y))[0]
        num_block_points = points_index.shape[0]
        num_block_points_list.append(num_block_points)

        count_block += 1
        # 将block的信息写入文件中
        utils.record_num_points(count_block, num_block_points, record_num_points_file, flag_x)
        utils.record_coordinates(count_block, start_x, start_y, end_x, end_y, record_coordinates_file, flag_x)
        # 将含有点云的block进行存储
        if num_block_points > 0:
            non_zero_block += 1
            block_data_file = os.path.join(train_blocks_path, "block_{}.txt".format(count_block))

            if format_flag:
                # 保留原始格式的方式进行存储
                f = open(block_data_file, "w")
                for i in range(num_block_points):
                    line_content = "{:.2f} {:.2f} {:.2f} {:d} {:d} {:d} {:d}\n".format(
                        train_data[points_index[i]][0], train_data[points_index[i]][1], train_data[points_index[i]][2],
                        int(train_data[points_index[i]][3]), int(train_data[points_index[i]][4]),
                        int(train_data[points_index[i]][5]), int(train_data[points_index[i]][6])
                    )
                    f.write(line_content)
                f.close()
            else:
                # 使用numpy方式进行存储，数据类型会变为np.float64
                np.savetxt(block_data_file, train_data[points_index])

            if num_block_points < threshold:
                less_threshold_list.append(count_block)

        """画图部分"""
        # 画出三维散点图
        ax3d.scatter(train_data[points_index][:, 0], train_data[points_index][:, 1], train_data[points_index][:, 2],
                     s=1, alpha=0.2)
        # 画出三维划分block的包裹框
        utils.plot_3d_linear_rect(ax3d, start_x, start_y, end_x, end_y,
                                  train_data[points_index][:, 0:3], default_z=(z_min, z_max))

        # 画出二维投影散点图
        ax2d.scatter(train_data[points_index][:, 0], train_data[points_index][:, 1], s=1, alpha=0.2)
        # 画出二维投影包裹框
        rect = mpathes.Rectangle((start_x, start_y), end_x - start_x, end_y - start_y, color="r", fill=False)
        ax2d.add_patch(rect)

        # 内循环已经超出边界了，则重置起始位置，并跳出内循环
        if flag_x:
            flag_x = False
            start_x = reset_x
            print("Inner loop is done!")
            break

        # 在内循环中进行步进
        start_x += stride_x

    # 在外循环已经超出边界了，则跳出外循环
    if flag_y:
        flag_y = False
        print("Outside loop is done!")
        break

    # 在外循环进行步进
    start_y += stride_y

fig2d.savefig(os.path.join(processed_path, "train2d.png"))
fig3d.savefig(os.path.join(processed_path, "train3d.png"))


print("None zero block: ", non_zero_block)
other_f.write("None zero block: {}\n".format(non_zero_block))
print("Total block: ", count_block)
other_f.write("Total block: {}\n".format(count_block))
print("Total points: ", sum(num_block_points_list))
other_f.write("Total points: {}\n".format(sum(num_block_points_list)))
print("Less threshold list: ", less_threshold_list)
other_f.write("Less threshold list: {}\n".format(less_threshold_list))

other_f.close()
print("Done!")
