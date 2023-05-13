from project_utils.cluster_utils import cluster_acc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np


def metrics_HNA(labels_true, labels_pre, unknown_babel):
    """
    :param labels_true:    ground-true open-set label   n*1
    :param labels_pre:     predicted open-set label     n*1
    :param unknown_babel:  label of unknown class = num_known_class + 1
    :return:               HNA
    """
    labels_true = labels_true.reshape(-1)
    index_known = np.where(labels_true != unknown_babel)
    index_unknown = np.where(labels_true == unknown_babel)
    AKS = accuracy_score(labels_true[index_known], labels_pre[index_known])      # AKS (accuracy of known classes)
    AUS = accuracy_score(labels_true[index_unknown], labels_pre[index_unknown])  # AUS (accuracy of unknown classes)
    if AKS == 0 or AUS == 0 or np.isnan(AKS) or np.isnan(AUS):
        HNA = 0
    else:
        HNA = 2 / (1 / AKS + 1 / AUS)

    return HNA


def metrics_OSFM(cm):
    """
    :param cm:  Confusion Matrix (num_known_class+1) * (num_known_class+1)
    :return:    Weighted-averaging open-set f-measure (OSFMw)
    """
    sum_rows, sum_cols = cm.sum(axis=1), cm.sum(axis=0)

    weights = normalization(sum_rows[:-1])
    f_measure_w = 0
    for i in range(cm.shape[0] - 1):
        tp_i = cm[i][i]
        fp_i = sum_cols[i]
        fn_i = sum_rows[i]
        precision_w = tp_i / np.maximum(fp_i, 0.001)
        recall_w = tp_i / np.maximum(fn_i, 0.001)
        f_measure_w += weights[i] * (2 * precision_w * recall_w) / np.maximum((precision_w + recall_w), 0.001)

    return f_measure_w


def normalization(data):
    """
    :param data:  matrix n*m
    :return:      convert each row of the matrix to the form of 1
    """
    if len(data.shape) == 2:
        _sum = (np.ones((data.shape[1], data.shape[0])) * np.sum(data, axis=1)).T
    else:
        _sum = np.sum(data)
    return data / _sum


def evaluation_closed(y_true, y_pred, phase, args):
    """
    :param y_true:  ground true label n*1
    :param y_pred:  predicted label n*1
    :param phase:   current stage
    :param args:    various parameters
    :print:         Accuracy
    """
    print("Performing {} closed-set recognition evaluation".format(phase))
    stage_name = ['1st', '2nd', '3rd', '4th']

    for p in range(int(phase[0])):
        if p == 0:
            condition = y_true < args.class_splits[p]
        else:
            condition = (args.class_splits[p-1] <= y_true) & (y_true < args.class_splits[p])

        acc = accuracy_score(y_true[condition], y_pred[condition])
        print("Accuracy on {} test set: {:.3f}".format(stage_name[p], acc))

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy on all test set: {:.3f}".format(acc))


def evaluation_open(y_true, y_pred, phase):
    """
    :param y_true:     ground-true open-set label   n*1
    :param y_pred:     predicted open-set label     n*1
    :param phase:      current stage
    :print:            HHA
    """
    print("Performing {} open-set recognition evaluation".format(phase))
    HNA = metrics_HNA(y_pred, y_true, np.max(y_true))
    print("HNA: {:.3f}".format(HNA))
    return HNA, None


def evaluation_novel(y_true, y_pred, phase, args):
    """
    The harmonic clustering accuracy is calculated
    :param y_true:       ground true label n*1
    :param y_pred:       predicted label n*1
    :param phase:        current stage
    :param args:         various parameters
    :print:              clustering accuracy on old, new, all classes
    """

    print("Performing {} generalized novel category discovery evaluation".format(phase))
    num_class_seen = args.class_splits[int(phase[0])-1]

    # Accuracy of seen classes
    condition_seen = y_true < num_class_seen
    acc_seen = accuracy_score(y_true[condition_seen], y_pred[condition_seen])

    condition_novel = (y_true >= num_class_seen) & (y_pred >= num_class_seen)
    if acc_seen == 0 or np.all(condition_novel == False):
        HCA = 0
    else:
        # Clustering accuracy of novel classes
        _, ind, w = cluster_acc(y_true[condition_novel], y_pred[condition_novel], return_ind=True)
        ind_novel = ind[ind[:, -1] >= num_class_seen]
        acc_novel = sum([w[i, j] for i, j in ind_novel]) * 1.0 / y_true[y_true >= num_class_seen].size
        # Harmonic clustering accuracy
        HCA = 2 / (1 / acc_seen + 1 / acc_novel)

    print("Harmonic clustering accuracy: {:.3f}".format(HCA))
