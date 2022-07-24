import numpy as np


def overall_accuracy(cm):
    """
    Compute the overall accuracy.
    :return: OA
    """
    return np.trace(cm) / cm.sum() * 100.0


def accuracy_per_class(cm):
    """
    Compute the accuracy per class and average
    puts -1 for invalid values (division per 0)
    returns average accuracy, accuracy per class
    :return:
        accuracy_per_class: ACC
        average_accuracy: ave ACC，mACC
    """
    sums = np.sum(cm, axis=1)
    mask = (sums > 0)
    sums[sums == 0] = 1
    accuracy_per_class = np.diag(cm) / sums
    accuracy_per_class[np.logical_not(mask)] = -1
    average_accuracy = accuracy_per_class[mask].mean()
    return average_accuracy * 100.0, accuracy_per_class * 100.0


def iou_per_class(cm, ignore_missing_classes=True):
    """
    Compute the iou per class and average iou
    Puts -1 for invalid values
    returns average iou, iou per class
    :return:
        iou_per_class: IoU
        average_iou: ave IoU，mIoU
    """
    sums = (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    mask = (sums > 0)
    sums[sums == 0] = 1
    iou_per_class = np.diag(cm) / sums
    iou_per_class[np.logical_not(mask)] = -1

    if mask.sum() > 0:
        average_iou = iou_per_class[mask].mean()
    else:
        average_iou = 0

    return average_iou * 100.0, iou_per_class * 100.0


def f1score_per_class(cm):
    """
    Compute f1 scores per class and mean f1.
    puts -1 for invalid classes
    returns average f1 score, f1 score per class
    F1 = 2*precision*recall / (precision+recall)
    precision = TP / (TP+FP), recall = TP / (TP+FN)
    F1 = 2*TP / (2*TP+FN+FP)
    :return:
        f1score_per_class: F1
        average_f1_score: avw F1，mF1
    """
    sums = (np.sum(cm, axis=1) + np.sum(cm, axis=0))
    mask = (sums > 0)
    sums[sums == 0] = 1
    f1score_per_class = 2 * np.diag(cm) / sums
    f1score_per_class[np.logical_not(mask)] = -1
    average_f1_score = f1score_per_class[mask].mean()
    return average_f1_score * 100.0, f1score_per_class * 100.0


def pfa_per_class(cm):
    """
    Compute the probability of false alarms.
    :return:
        pfa_per_class:
        average_pfa
    """
    sums = np.sum(cm, axis=0)
    mask = (sums > 0)
    sums[sums == 0] = 1
    pfa_per_class = (cm.sum(axis=0) - np.diag(cm)) / sums
    pfa_per_class[np.logical_not(mask)] = -1
    average_pfa = pfa_per_class[mask].mean()
    return average_pfa, pfa_per_class

