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
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


from model.function_utils import *
import data_utils.d_utils as d_utils


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

train_label_weights = [546, 180850, 193723, 4614, 12070, 152045, 27250, 47605, 135173]
val_label_weights = [600, 98690, 101986, 3708, 7422, 109048, 11224, 24818, 54226]
total_label_weights = [1146, 279540, 295709, 8322, 19492, 261093, 38474, 72423, 189399]


def convert_label_weights(label_weights):
    label_weights = np.array(label_weights)
    label_weights = label_weights.astype(np.float32)
    label_weights = label_weights / np.sum(label_weights)
    label_weights = np.power(np.amax(label_weights) / label_weights, 1 / 3.0)
    return torch.tensor(label_weights)

train_label_weights = convert_label_weights(train_label_weights)


parser = argparse.ArgumentParser(description="isprs")
# data params
parser.add_argument("--num_classes", type=int, default=9)
parser.add_argument("--batch_size", type=int, default=16, help="Batch size [default: 32]")
parser.add_argument("--num_points", type=int, default=4096, help="Number of points to train with [default: 4096]")
parser.add_argument("--input_channels", type=int, default=5)
parser.add_argument("--use_xyz", type=bool, default=True)
parser.add_argument("--isprs_root", type=str, default="/workspace/RSPointCloud/Vaihingen3D")
# training params
parser.add_argument("--lr", type=float, default=0.002, help="Initial learning rate [default: 1e-2]")
parser.add_argument('--lr_min', type=float, default=1e-5, help='min learning rate')
parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 regularization coeff [default: 0.0]")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train for")
# other params
parser.add_argument("--seed", type=int, default=166189)
parser.add_argument("--save", type=str, default="./RFFSNet")
parser.add_argument("--train_report_freq", type=int, default=10)
parser.add_argument("--infer_report_freq", type=int, default=200)


def save_parms(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_model(model, model_path):
    torch.save(model, model_path)


def load_parms(model, model_path):
    model.load_state_dict(torch.load(model_path))


def load_model(model_path):
    return torch.load(model_path)


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


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.save = '{}-{}-{}'.format(args.save, "train", time.strftime("%Y%m%d-%H%M%S"))

    if not os.path.exists(args.save):
        os.mkdir(args.save)
    shutil.copy("train.py", args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    print('Experiment_dir: {}'.format(args.save))
    logging.info("Directory has been created!")

    train_merge_block_path = osp.join(args.isprs_root, "processed_no_rgb", "train_merge")
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    train_set = ISPRS3DDataset(args.num_points, train_merge_block_path, train_transforms)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, num_workers=4, shuffle=True)
    logging.info("Train data have been loaded!")

    test_merge_block_path = osp.join(args.isprs_root, "processed_no_rgb", "eval_merge")
    test_set = ISPRS3DDataset(args.num_points, test_merge_block_path, None)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=4, shuffle=False)
    logging.info("Test data have been loaded!")

    model = RFFSNet(input_channels=5, num_classes=9)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = lr_sched.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min)

    criterion = nn.CrossEntropyLoss(weight=train_label_weights)
    criterion = criterion.cuda()

    logging.info("args = %s", args)

    best_mf1 = 0.0
    for epoch in range(args.epochs):
        lr_scheduler.step()
        logging.info("**********epoch: %03d**********", epoch)
        logging.info("lr %e", lr_scheduler.get_lr()[0])

        logging.info(">>>Training>>>")
        train_oa, train_macc, train_miou, train_mf1 = train(args, model, train_loader, optimizer, criterion, logging)
        logging.info(">>>Infer>>>")
        infer_oa, infer_macc, infer_iou, infer_mf1 = infer(args, model, test_loader, criterion, logging)

        if infer_mf1 > best_mf1:
            best_mf1 = infer_mf1
            save_parms(model, osp.join(args.save, 'params.pt'))
            save_model(model, osp.join(args.save, 'models.pt'))
            logging.info('the model in epoch %d is the best', epoch)


def train(args, model, train_loader, optimizer, criterion, logging):
    objs = AvgrageMeter()

    model.train()
    train_cm = np.zeros((args.num_classes, args.num_classes))
    for step, batch in enumerate(train_loader):
        inputs, labels = batch
        inputs = inputs.to("cuda", non_blocking=True)
        labels = labels.to("cuda", non_blocking=True)

        optimizer.zero_grad()
        preds, decoder_out, decoder_labels = model(inputs, labels)
        loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))
        loss1 = criterion(decoder_out[0].contiguous().view(decoder_labels[2].contiguous().numel(), -1),
                          decoder_labels[2].contiguous().view(-1))
        loss2 = criterion(decoder_out[1].contiguous().view(decoder_labels[1].contiguous().numel(), -1),
                          decoder_labels[1].contiguous().view(-1))
        loss3 = criterion(decoder_out[2].contiguous().view(decoder_labels[0].contiguous().numel(), -1),
                          decoder_labels[0].contiguous().view(-1))
        loss = loss + 0.3 * loss1 + 0.3 * loss2 + 0.3 * loss3
        loss.backward()
        optimizer.step()

        preds_np = np.argmax(preds.cpu().detach().numpy(), axis=2).copy()
        labels_np = labels.cpu().numpy().copy()

        train_cm_ = confusion_matrix(labels_np.ravel(), preds_np.ravel(), labels=list(range(args.num_classes)))
        train_cm += train_cm_

        objs.update(loss.item(), inputs.shape[0] * args.num_points)
        if step % args.train_report_freq == 0:
            logging.info('Train Step: %04d Loss: %f', step, objs.avg)

    train_oa, train_macc, train_miou, train_mf1 = calulate_output_metrics(train_cm, args.num_classes, logging)

    return train_oa, train_macc, train_miou, train_mf1


def infer(args, model, test_loader, criterion, logging):
    objs = AvgrageMeter()

    model.eval()
    val_cm = np.zeros((args.num_classes, args.num_classes))
    for step, batch in enumerate(test_loader):
        try:
            inputs, labels, _ = batch
        except:
            inputs, labels = batch
        inputs = inputs.to("cuda", non_blocking=True)
        labels = labels.to("cuda", non_blocking=True)

        with torch.no_grad():
            preds, _, _ = model(inputs, labels)

        loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))

        preds_np = np.argmax(preds.cpu().detach().numpy(), axis=2).copy()
        labels_np = labels.cpu().numpy().copy()

        val_cm_ = confusion_matrix(labels_np.ravel(), preds_np.ravel(), labels=list(range(args.num_classes)))
        val_cm += val_cm_

        objs.update(loss.item(), inputs.shape[1])
        if step % args.infer_report_freq == 0:
            logging.info('Infer Step: %04d Loss: %f', step, objs.avg)

    infer_oa, infer_macc, infer_iou, infer_mf1 = calulate_output_metrics(val_cm, args.num_classes, logging)

    return infer_oa, infer_macc, infer_iou, infer_mf1


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("Done!")
