from __future__ import division
from __future__ import with_statement
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import sys
import time
import torch
import shutil
import logging
import argparse
import numpy as np
import torch.nn as nn
import os.path as osp

from model.function_utils import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from model.rffsnet import RFFSNet
from data_utils.ISPRS3DDataLoader import ISPRS3DDataset, ISPRS3DWholeDataset


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class2label = {
    'Powerline': 0, 'Low_vegetation': 1, 'Impervious_surfaces': 2, 'Car': 3, 'Fence': 4,
    'Roof': 5, 'Facade': 6, 'Shrub': 7, 'Tree': 8
}
label2class = {
    0: 'Powerline', 1: 'Low_vegetation', 2: 'Impervious_surfaces', 3: 'Car', 4: 'Fence',
    5: 'Roof', 6: 'Facade', 7: 'Shrub', 8: 'Tree'
}

# class2rgb = {
#     0: [255, 255, 125], 1: [0, 255, 255], 2: [255, 255, 255], 3: [255, 255, 0], 4: [0, 255, 125],
#     5: [0, 0, 255], 6: [0, 125, 255], 7: [125, 255, 0], 8: [0, 255, 0]
# }

class2rgb = {
    0: [255, 105, 180],      # Powerline: hot pink
    1: [170, 255, 127],      # Low vegetation:
    2: [128, 128, 128],      # Impervious surfaces: gray
    3: [255, 215, 0],        # Car: gold
    4: [0, 191, 255],        # Fence: deep sky blue
    5: [0, 0, 127],          # Roof: blue
    6: [205, 133, 0],        # Facade: orange
    7: [160, 32, 240],       # Shrub: purple
    8: [9, 120, 26],         # Tree: green
}

parser = argparse.ArgumentParser(description="isprs")
# data params
parser.add_argument("--num_classes", type=int, default=9)
parser.add_argument("--batch_size", type=int, default=8, help="Batch size [default: 32]")
parser.add_argument("--num_points", type=int, default=4096, help="Number of points to train with [default: 4096]")
parser.add_argument("--input_channels", type=int, default=5)
parser.add_argument("--use_xyz", type=bool, default=True)
parser.add_argument("--isprs_root", type=str, default="/workspace/RSPointCloud/Vaihingen3D")
parser.add_argument("--num_votes", type=int, default=1)
# parser.add_argument("--num_votes", type=int, default=10)
# other params
parser.add_argument("--seed", type=int, default=166189)
parser.add_argument('--model_path', type=str, default='/workspace/RFFSNet/tools/RFFSNet-train-20220724-103622/models.pt')
# parser.add_argument('--param_path', type=str, default='/workspace/PointConv/ISPRS-train-20160215-164519/params.pt')
parser.add_argument("--save", type=str, default="./ISPRStest")
parser.add_argument("--infer_report_freq", type=int, default=50)


def save_parms(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_model(model, model_path):
    torch.save(model, model_path)


def load_parms(model, model_path):
    model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load(model_path)["model_state"])


def load_model(model_path):
    return torch.load(model_path)


def preds2rgb(preds_np):
    """
    :param preds_np:
    :return: 
    """
    num_points = preds_np.size
    map_rgb_np = np.zeros((num_points, 3))
    for i, ele in enumerate(preds_np):
        map_rgb_np[i] = np.array(class2rgb[ele])
    return map_rgb_np


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def model_vote(num_points, inputs, labels, model, vote_preds):
    """
    :param num_points:
    :param inputs: 
    :param model: 
    :param vote_preds:
    :return: 
    """
    block_num_points = inputs.shape[1]
    num_batches = int(np.ceil(block_num_points / num_points))

    points_size = int(num_batches * num_points)

    replace = False if (points_size - block_num_points <= block_num_points) else True

    point_idxs_repeat = np.random.choice(block_num_points, points_size - block_num_points, replace=replace)
    point_idxs = np.concatenate((range(block_num_points), point_idxs_repeat))
    np.random.shuffle(point_idxs)

    for i in range(num_batches):
        current_idxs = point_idxs[i*num_points:(i+1)*num_points]
        with torch.no_grad():
            logits, _, _ = model(inputs[:, current_idxs.tolist(), :], labels)
        preds = torch.argmax(logits, dim=2)
        vote_preds[current_idxs, preds] += 1
    return vote_preds


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", args)

    print('Experiment_dir: {}'.format(args.save))
    logging.info("Directory has been created!")

    test_merge_block_path = osp.join(args.isprs_root, "processed_no_rgb", "eval_merge")
    # test_set = ISPRS3DDataset(args.num_points, test_merge_block_path, None)
    test_set = ISPRS3DWholeDataset(test_merge_block_path, None)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=4, shuffle=False)
    logging.info("Test data have been loaded!")

    model = RFFSNet(input_channels=5, num_classes=9)
    # load_parms(model, args.param_path)
    model = model.cuda()

    model = load_model(args.model_path)
    # model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    all_points = []

    model.eval()
    objs = AvgrageMeter()
    val_cm = np.zeros((args.num_classes, args.num_classes))
    for step, data in enumerate(test_loader):
        # inputs, labels = data
        inputs, labels, block_path = data
        inputs = inputs.to("cuda", non_blocking=True)
        labels = labels.to("cuda", non_blocking=True)

        if args.num_votes > 0:
            vote_preds = torch.zeros((inputs.shape[1], args.num_classes)).to("cuda", non_blocking=True)
            for _ in range(args.num_votes):
                vote_preds = model_vote(args.num_points, inputs, labels, model, vote_preds)
            final_preds = vote_preds
        else:
            with torch.no_grad():
                final_preds = model(inputs)
            final_preds = final_preds[0]

        loss = criterion(final_preds.view(labels.numel(), -1), labels.view(-1))

        preds_np = np.argmax(final_preds.cpu().detach().numpy(), axis=1).copy()
        labels_np = labels.cpu().numpy().copy()

        val_cm_ = confusion_matrix(labels_np.ravel(), preds_np.ravel(), labels=list(range(args.num_classes)))
        val_cm += val_cm_

        objs.update(loss.item(), inputs.shape[1])
        if step % args.infer_report_freq == 0:
            logging.info('Infer Step: %04d Loss: %f', step, objs.avg)

    calulate_output_metrics(val_cm, args.num_classes, logging)


def calulate_output_metrics(confusion_matrix, num_classes, logging):
    """
    :param confusion_matrix:
    :param num_classes: 
    :param logging: 
    :return: 
    """
    oa = overall_accuracy(confusion_matrix)
    macc, acc = accuracy_per_class(confusion_matrix)
    miou, iou = iou_per_class(confusion_matrix)
    mf1, f1 = f1score_per_class(confusion_matrix)

    logging.info("Total Results")
    logging.info("OA: {:9f}, mACC: {:9f}, mIoU: {:9f}, mF1: {:9f}".format(oa, macc, miou, mf1))
    logging.info("Each class Results")
    for i in range(num_classes):
        message = "{:19} Result: ACC {:9f}, IoU {:9f}, F1 {:9f}".format(label2class[i], acc[i], iou[i], f1[i])
        logging.info(message)

    return oa, macc, miou, mf1


if __name__ == "__main__":
    args = parser.parse_args()
    args.save = '{}-{}-{}'.format(args.save, "test", time.strftime("%Y%m%d-%H%M%S"))
    main(args)
    print("Eval done!")
