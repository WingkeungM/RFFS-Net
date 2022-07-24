import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree
import data_utils.d_utils as d_utils


class ISPRS3DDataset(Dataset):
    def __init__(self, num_points, merge_block_path, data_transforms):
        self.num_points = num_points
        self.merge_block_path = merge_block_path
        self.data_transforms = data_transforms

        block_num_points = []
        self.blocks_path = os.listdir(merge_block_path)
        for block_path in self.blocks_path:
            block_data = np.loadtxt(os.path.join(merge_block_path, block_path))
            block_num_points.append(block_data.shape[0])

        sample_prob = block_num_points / np.sum(block_num_points)
        num_iter = int(np.sum(block_num_points) / num_points)
        batch_idxs = []

        for block_index in range(len(self.blocks_path)):
            batch_idxs.extend([self.blocks_path[block_index]] * int(np.ceil(sample_prob[block_index] * num_iter)))
        self.batch_idxs = batch_idxs

        self.x_min, self.x_max = 496848.61, 497231.47
        self.y_min, self.y_max = 5419179.21, 5419584.5
        self.z_min, self.z_max = 250.6, 289.91

        self.x_interval = self.x_max - self.x_min
        self.y_interval = self.y_max - self.y_min
        self.z_interval = self.z_max - self.z_min

    def __len__(self):
        return len(self.batch_idxs)

    def __getitem__(self, idx):
        block_path = os.path.join(self.merge_block_path, self.batch_idxs[idx])
        block_data = np.loadtxt(block_path)
        all_points_idxs = block_data.shape[0]

        if all_points_idxs < self.num_points:
            selected_points_idxs = np.random.choice(all_points_idxs, self.num_points, replace=True)
        else:
            seed = np.random.choice(all_points_idxs, 1)
            center_point = block_data[seed][:, 0:3]
            search_tree = KDTree(block_data[:, 0:3], leaf_size=10)
            _, ind = search_tree.query(center_point, k=self.num_points)
            selected_points_idxs = ind[0]

        points = block_data[selected_points_idxs, 0:3].astype(np.float32)
        intensity = block_data[selected_points_idxs, 3].astype(np.float32)
        labels = block_data[selected_points_idxs, 6].astype(np.int64)

        current_points = np.ones((self.num_points, 8), dtype=np.float32)        # XYZxyzI1
        current_points[:, 0:3] = points

        center = np.mean(points, axis=0)[:3]
        current_points[:, 0:3] -= center

        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        z_min = np.min(points[:, 2])
        i_min = np.min(intensity)
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])
        z_max = np.max(points[:, 2])
        i_max = np.max(intensity)
        current_points[:, 3] = (points[:, 0] - x_min) / (x_max - x_min)
        current_points[:, 4] = (points[:, 1] - y_min) / (y_max - y_min)
        current_points[:, 5] = (points[:, 2] - z_min) / (z_max - z_min)
        current_points[:, 6] = (intensity - i_min) / (i_max - i_min)

        inputs = current_points

        return inputs, labels


class ISPRS3DWholeDataset(Dataset):
    def __init__(self, merge_block_path, data_transforms):
        self.merge_block_path = merge_block_path
        self.data_transforms = data_transforms
        self.blocks_path = os.listdir(merge_block_path)

        self.x_min, self.x_max = 497073.33, 497446.9
        self.y_min, self.y_max = 5419592.78, 5419995.37
        self.z_min, self.z_max = 265.44, 296.85

        self.x_interval = self.x_max - self.x_min
        self.y_interval = self.y_max - self.y_min
        self.z_interval = self.z_max - self.z_min

    def __len__(self):
        return len(self.blocks_path)

    def __getitem__(self, idx):
        block_path = os.path.join(self.merge_block_path, self.blocks_path[idx])
        block_data = np.loadtxt(block_path)
        points = block_data[:, 0:3].astype(np.float32)
        intensity = block_data[:, 3].astype(np.float32)
        labels = block_data[:, 6].astype(np.int64)

        current_points = np.ones((block_data.shape[0], 8), dtype=np.float32)        # XYZxyzI1
        current_points[:, 0:3] = points

        center = np.mean(points, axis=0)[:3]
        current_points[:, 0:3] -= center

        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        z_min = np.min(points[:, 2])
        i_min = np.min(intensity)
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])
        z_max = np.max(points[:, 2])
        i_max = np.max(intensity)
        current_points[:, 3] = (points[:, 0] - x_min) / (x_max - x_min)
        current_points[:, 4] = (points[:, 1] - y_min) / (y_max - y_min)
        current_points[:, 5] = (points[:, 2] - z_min) / (z_max - z_min)

        current_points[:, 6] = (intensity - i_min) / (i_max - i_min)
        inputs = current_points

        return inputs, labels, block_path


if __name__ == "__main__":
    # import random
    # random.seed(0)
    # np.random.seed(0)
    #
    # isprs_root = "/dataset/Vaihingen3D"
    # merge_block_path = os.path.join(isprs_root, "processed_no_rgb", "train_merge")
    # num_points = 4096
    #
    # from torchvision import transforms
    # train_transforms = transforms.Compose([
    #     data_utils.PointcloudToTensor(),
    #     # data_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
    #     # data_utils.PointcloudScale(),
    #     # data_utils.PointcloudTranslate(),
    #     # data_utils.PointcloudJitter(),
    # ])
    # isprs3d = ISPRS3DDataset(num_points, merge_block_path, train_transforms)
    #
    # from torch.utils.data import DataLoader
    # train_loader = DataLoader(isprs3d, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    # for i, (inputs, labels) in enumerate(train_loader):
    #     print(inputs.shape[0])
    #
    # print("Done!")

    isprs_root = "/dataset/Vaihingen3D"
    merge_block_path = os.path.join(isprs_root, "processed_no_rgb", "eval_merge")
    num_classes = 9
    normal = "local"

    wholeisprs = ISPRS3DWholeDataset(merge_block_path, data_transforms=None)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(wholeisprs, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    for i, (points, labels, block_path) in enumerate(test_loader):
        print(points.shape[1])
    print("Done!")
