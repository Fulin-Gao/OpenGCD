from sklearn.feature_selection import mutual_info_regression
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def map01_to_ab(a, b, v):
    """
    :param a:    Lower bound after mapping
    :param b:    Upper bound after mapping
    :param v:    Vector to be mapped
    :return:     Mapped vector range [a, b]
    """
    # Map v to [a, b]
    v_max = np.max(v)
    v_min = np.min(v)
    k = (b - a) / (v_max - v_min)
    y = a + k * (v - v_min)
    return y


def distance(A, B):
    """
    :param A:     Matrix A
    :param B:     Matrix B
    :return:      The distance matrix between two pairs
    """
    step0 = -2 * np.dot(A, B.T) + np.sum(np.square(B), axis=1) + np.transpose([np.sum(np.square(A), axis=1)])
    step0[step0 <= 0] = 0
    dists = np.sqrt(step0)
    return dists


def mic(train_feats_exemplar, train_targets_exemplar, num_known_class):
    """
    :param train_feats_exemplar:           Raw features
    :param train_targets_exemplar:         Raw targets
    :param num_known_class:                Number of known classes
    :return:                               Softened labels
    """
    # Define the prototype
    prototypes = np.zeros((num_known_class, train_feats_exemplar.shape[1]))
    for c in range(num_known_class):
        prototypes[c] = np.mean(train_feats_exemplar[train_targets_exemplar == c], axis=0)

    # Calculating normalized mutual information
    mis = np.zeros((train_feats_exemplar.shape[0], num_known_class))
    for i, x in enumerate(tqdm(train_feats_exemplar)):
        for j, p in enumerate(prototypes):
            mis[i, j] = mutual_info_regression(x.reshape(-1, 1), p)

    # find nearest neighbor class and map nmi to (0.5, 1] for each class
    nmi_norm = -1 * np.ones(len(train_targets_exemplar))
    idx_nn = -1 * np.ones(len(train_targets_exemplar))
    for c in range(num_known_class):
        condition = train_targets_exemplar == c
        mis_c = mis[condition]

        # find nearest neighbor class
        idx_second_max = np.argsort(mis_c, axis=1)[:, -2]  # second max index of row
        idx_max = np.argmax(mis_c, axis=1)  # max index of row
        idx_error = np.where(idx_second_max == c)
        if len(idx_error) != 0:
            idx_second_max[idx_error] = idx_max[idx_error]
        idx_nn[condition] = idx_second_max

        # map nmi to (0.5, 1]
        mi_c = mis_c[:, c]
        mi_c_norm = map01_to_ab(0.5+1e-2, 1, mi_c)  # 0.99
        nmi_norm[condition] = mi_c_norm

    # Assigning soft labels
    gt_target_onehot = tf.one_hot(train_targets_exemplar, depth=num_known_class).numpy()
    nn_target_onehot = tf.one_hot(idx_nn, depth=num_known_class).numpy()
    target_soft = np.multiply(nmi_norm, gt_target_onehot.T).T + np.multiply((1-nmi_norm), nn_target_onehot.T).T
    return target_soft


if __name__ == "__main__":
    from sklearn import datasets

    iris = datasets.load_digits()
    irisFeatures = iris["data"]
    irisLabels = iris["target"]

    target_soft = mic(irisFeatures, irisLabels, num_known_class=10)

    print("")
