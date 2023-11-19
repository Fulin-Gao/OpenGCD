from sklearn.feature_selection import mutual_info_regression
from methods.exemplars_selection.DS3 import DS3
from scipy.special import softmax
import numpy as np
import time
import math
import warnings

warnings.filterwarnings("ignore")


def distance_matrix(A, B):
    """
    :param A:  matrix A n1*m
    :param B:  matrix B n2*m
    :return:   distance matrix between two pairs
    """
    step0 = -2 * np.dot(A, B.T) + np.sum(np.square(B), axis=1) + np.transpose([np.sum(np.square(A), axis=1)])
    step0[step0 <= 0] = 0
    dists = np.sqrt(step0)
    return dists


def runDS3(D, reg, k, verbose=False):
    """
            This function runs DS3.
            :param D:    matrix whose non-zero rows corresponds to the representatives of the dataset.
            :param reg:  norm to be used to calculate regularization cost.
            :param k:    retain variable number
            :returns:    regularization cost.
    """
    # initialize DS3 class with dis-similarity matrix and the regularization parameter.
    dis_matrix = D
    DS = DS3(dis_matrix, reg)

    # run the ADMM algorithm.
    start = time.time()
    data_admm, num_of_rep_admm = \
        DS.ADMM(mu=10 ** -1, epsilon=10 ** -7, max_iter=10, p=np.inf, k=k)
    end = time.time()
    rep_super_frames = data_admm

    # change the above indices into 0s and 1s for all indices.
    N = len(D)
    summary = np.zeros(N)
    for i in range(len(rep_super_frames)):
        summary[rep_super_frames[i]] = 1

    run_time = end - start
    idx = []
    for index, i in enumerate(summary):
        if i == 1:
            idx.append(index)

    idx = np.asarray(idx)
    if verbose:
        print("Run Time :", run_time)
        print("Summary :", summary)
        print("Index representative :", idx)

    return idx


def sampling_byDS3(all_known_feats, all_known_targets, num_known_class, r, k, phase, args, model=None):
    """
    :param all_known_feats:    features
    :param all_known_targets:  labels
    :param num_known_class:    number of known classes
    :param r:                  adjustment factor for the number of samples
    :param k:                  desired number of reservations
    :param phase:              current phase
    :param args:               all args
    :param args:               fitted model
    :return:                   preserved features and corresponding labels
    """
    memory_c = np.zeros(num_known_class, dtype=int)

    if phase == "1st":
        num_old_class = 0
        memory_c += k * np.ones(num_known_class, dtype=int)
    else:
        num_old_class = args.class_splits[int(phase[0]) - 2]

        # Define the prototype and calculate normalized mutual information
        mis_c = []
        for c in range(num_known_class):
            condition = all_known_targets == c
            c_known_feats = all_known_feats[condition]
            p = np.mean(c_known_feats, axis=0)
            mi = mutual_info_regression(c_known_feats.T, p)
            mis_c.append(np.mean(mi))

        # Allocate memory
        mis_norm = softmax(np.array(mis_c))
        for c in range(num_known_class):
            memory_c[c] = mis_norm[c] * args.memory

    # Sampling
    new_feats = []
    new_labels = []
    length_per_class = []
    for c in range(num_known_class):
        idx = np.where(all_known_targets == c)[0]
        if c < num_old_class:
            new_feats += [all_known_feats[idx][:memory_c[c]]]
            new_labels += [all_known_targets[idx][:memory_c[c]]]
            length_per_class.append(memory_c[c])
        else:
            dis_matrix = distance_matrix(all_known_feats[idx, :], all_known_feats[idx, :])
            idx_byDS3 = runDS3(dis_matrix, r, memory_c[c])
            retain_x, retain_y = all_known_feats[idx[idx_byDS3], :], all_known_targets[idx[idx_byDS3]]
            new_feats = new_feats + [retain_x]
            new_labels = new_labels + [retain_y]
            length_per_class.append(len(idx_byDS3))
    new_feats = np.concatenate(new_feats)
    new_labels = np.concatenate(new_labels)
    print('Number of exemplars retained per class for {p}: {n}'.format(p=phase, n=length_per_class))
    return new_feats, new_labels
