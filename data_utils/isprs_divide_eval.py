"""将ISPRS 3D测试集中的两个区域分隔开，两个区域的分界点为179997"""

import utils
import numpy as np
import matplotlib.pyplot as plt


format_flag = False
print("保留原始格式进行存储：", format_flag)
x_mid = 497249.5        # 497245.5 - 497251.55
y_mid = 5419733.5       # 5419733.0 - 5419734.5

isprs_eval = "/dataset/Vaihingen3D/Vaihingen3D_EVAL_WITH_REF.pts"
data = np.loadtxt(isprs_eval)
num_points = data.shape[0]
points_xyz = data[:, 0:3]
# x_min: 497073.33, x_max: 497446.9; y_min: 5419592.78, y_max: 5419995.37; z_min: 265.44, z_max: 296.85
x_min, x_max, y_min, y_max, z_min, z_max = utils.find_min_max(points_xyz)

# 分离出测试集中的区域1
eval1_points_index = np.where((points_xyz[:, 0] >= x_min) & (points_xyz[:, 0] <= x_mid) &
                              (points_xyz[:, 1] >= y_mid) & (points_xyz[:, 1] <= y_max))[0]
eval1_block_points = points_xyz[eval1_points_index]
num_eval1_block_points = eval1_block_points.shape[0]
print("区域1点云数目：", num_eval1_block_points)

# 分离出测试集中的区域2
eval2_points_index = np.where(~((points_xyz[:, 0] >= x_min) & (points_xyz[:, 0] <= x_mid) &
                                (points_xyz[:, 1] >= y_mid) & (points_xyz[:, 1] <= y_max)))[0]
eval2_block_points = points_xyz[eval2_points_index]
num_eval2_block_points = eval2_block_points.shape[0]
print("区域2点云数目：", num_eval2_block_points)

print("区域1和区域2的总点云数目：", num_eval1_block_points + num_eval2_block_points)


# 保留两位小数进行存储，或直接保存为np.float64进行存储
if format_flag:
    test1_txt = "/dataset/Vaihingen3D/eval_with_ref1_raw_format.txt"
    test2_txt = "/dataset/Vaihingen3D/eval_with_ref2_raw_format.txt"

    # 保留原始格式的方式进行存储
    f = open(test1_txt, "w")
    for i in range(num_eval1_block_points):
        line_content = "{:.2f} {:.2f} {:.2f} {:d} {:d} {:d} {:d}\n".format(
            data[eval1_points_index[i]][0], data[eval1_points_index[i]][1], data[eval1_points_index[i]][2],
            int(data[eval1_points_index[i]][3]), int(data[eval1_points_index[i]][4]),
            int(data[eval1_points_index[i]][5]), int(data[eval1_points_index[i]][6])
        )
        f.write(line_content)
    f.close()

    f = open(test2_txt, "w")
    for i in range(num_eval1_block_points):
        line_content = "{:.2f} {:.2f} {:.2f} {:d} {:d} {:d} {:d}\n".format(
            data[eval2_points_index[i]][0], data[eval2_points_index[i]][1], data[eval2_points_index[i]][2],
            int(data[eval2_points_index[i]][3]), int(data[eval2_points_index[i]][4]),
            int(data[eval2_points_index[i]][5]), int(data[eval2_points_index[i]][6])
        )
        f.write(line_content)
    f.close()
else:
    test1_txt = "/dataset/Vaihingen3D/eval_with_ref1.txt"
    test2_txt = "/dataset/Vaihingen3D/eval_with_ref2.txt"

    # 使用numpy方式进行存储，数据类型会变为np.float64
    np.savetxt(test1_txt, data[eval1_points_index])
    np.savetxt(test2_txt, data[eval2_points_index])

print("Done!")

# 画图检验区域1
fig = plt.figure(1)
plt.scatter(eval1_block_points[:, 0], eval1_block_points[:, 1])
plt.savefig("eval1.png")

fig = plt.figure(2)
plt.scatter(eval2_block_points[:, 0], eval2_block_points[:, 1])
plt.savefig("eval2.png")
