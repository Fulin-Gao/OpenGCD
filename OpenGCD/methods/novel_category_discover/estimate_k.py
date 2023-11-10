from methods.novel_category_discover.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import davies_bouldin_score
from sklearn.metrics.cluster import silhouette_score
from project_utils.cluster_utils import cluster_acc
from scipy.optimize import minimize_scalar
from functools import partial
from sko.GA import GA
from sko.DE import DE
import numpy as np
import torch
import math


def ss_kmeans(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, K, args):
    """
    :param labeled_feats:        labeled data features    n1*m
    :param labeled_targets:      labeled data labels      n1*1
    :param unlabeled_feats:      unlabeled data features  n2*m
    :param num_known_class:      number of known classes
    :param K:                    total classes number
    :param args:                 various parameters
    :return:                     semi-supervised performance indicators average clustering accuracy (ACC) and Davies Bouldin score (DBS)
    """
    # Split labeled data into an anchor probe set and a validation probe set (2:1)
    num_class_anchor = math.ceil(2 * num_known_class / 3)
    mask_probe = np.zeros(len(labeled_targets), dtype=int)  # {anchor:0, validation:1}
    mask_probe[labeled_targets >= num_class_anchor] = 1
    anchor_feats = labeled_feats[mask_probe == 0]
    val_feats = labeled_feats[mask_probe == 1]
    anchor_targets = labeled_targets[mask_probe == 0]
    val_targets = labeled_targets[mask_probe == 1]

    # Merge validation probe set and unknown unlabeled data
    u_feats = np.concatenate((val_feats, unlabeled_feats), axis=0)
    mask = np.concatenate((mask_probe, 2 * np.ones(unlabeled_feats.shape[0], dtype=int)))  # {anchor:0, validation:1, unknown unlabeled:2}

    # Perform semi-supervised k-means clustering
    print('Performing semi-supervised k-means (k={})'.format(K))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    l_feats, u_feats, l_targets = (torch.from_numpy(x).to(device) for x in
                                              (anchor_feats, u_feats, anchor_targets))
    ss_k_means = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                               mode=None)
    ss_k_means.fit_mix(u_feats, l_feats, l_targets)

    # Get predictions corresponding to validation probe set and unknown unlabeled data
    all_pred = ss_k_means.labels_.cpu().numpy()
    val_pred = all_pred[mask == 1]
    unlabeled_pred = all_pred[mask == 2]

    # Evaluate clustering results
    ACC = fowlkes_mallows_score(val_targets, val_pred)
    DBS = davies_bouldin_score(unlabeled_feats, unlabeled_pred)
    return ACC, DBS


def estimate_k(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, phase, args):
    """
    :param labeled_feats:        labeled data features    n1*m
    :param labeled_targets:      labeled data labels      n1*1
    :param unlabeled_feats:      unlabeled data features  n2*m
    :param num_known_class:      number of known classes
    :param phase:                current phase
    :param args:                 various parameters
    :return:                     best total number of categories K
    """
    records_ACC = {}
    records_DBS = {}
    for K in range(num_known_class+1, args.max_K+1):  # There is at least 1 novel class, and at most the total number of classes is max_K
        ACC, DBS = ss_kmeans(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, K, args)
        records_ACC[K] = ACC
        records_DBS[K] = DBS
    best_K_ACC = max(records_ACC, key=lambda k: records_ACC[k])
    best_K_DBS = max(records_DBS, key=lambda k: records_DBS[k])
    best_K = math.ceil((best_K_ACC + best_K_DBS) / 2)
    print(records_ACC)
    print(records_DBS)
    print("The best K is {} for {}".format(best_K, phase))
    return best_K


def ss_kmeans_for_search(K, labeled_feats, labeled_targets, unlabeled_feats, num_known_class, args):
    """
    :param labeled_feats:        labeled data features    n1*m
    :param labeled_targets:      labeled data labels      n1*1
    :param unlabeled_feats:      unlabeled data features  n2*m
    :param num_known_class:      number of known classes
    :param K:                    total classes number
    :param args:                 various parameters
    :return:                     semi-supervised performance indicators average clustering accuracy (ACC) and Davies Bouldin score (DBS)
    """
    # Split labeled data into an anchor probe set and a validation probe set (2:1)
    num_class_anchor = math.ceil(2 * num_known_class / 3)
    mask_probe = np.zeros(len(labeled_targets), dtype=int)  # {anchor:0, validation:1}
    mask_probe[labeled_targets >= num_class_anchor] = 1
    anchor_feats = labeled_feats[mask_probe == 0]
    val_feats = labeled_feats[mask_probe == 1]
    anchor_targets = labeled_targets[mask_probe == 0]
    val_targets = labeled_targets[mask_probe == 1]

    # Merge validation probe set and unknown unlabeled data
    u_feats = np.concatenate((val_feats, unlabeled_feats), axis=0)
    mask = np.concatenate((mask_probe, 2 * np.ones(unlabeled_feats.shape[0], dtype=int)))  # {anchor:0, validation:1, unknown unlabeled:2}

    # Perform semi-supervised k-means clustering
    K = int(K)
    print('Performing semi-supervised k-means (k={})'.format(K))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    l_feats, u_feats, l_targets = (torch.from_numpy(x).to(device) for x in
                                              (anchor_feats, u_feats, anchor_targets))
    ss_k_means = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                               mode=None)
    ss_k_means.fit_mix(u_feats, l_feats, l_targets)

    # Get predictions corresponding to validation probe set and unknown unlabeled data
    all_pred = ss_k_means.labels_.cpu().numpy()
    val_pred = all_pred[mask == 1]
    unlabeled_pred = all_pred[mask == 2]

    # Evaluate clustering results
    ACC = fowlkes_mallows_score(val_targets, val_pred)  # fowlkes_mallows_score(val_targets, val_pred)  cluster_acc
    DBS = davies_bouldin_score(unlabeled_feats, unlabeled_pred)  # davies_bouldin_score(unlabeled_feats, unlabeled_pred)  # silhouette_score
    print('K:{:.2f}, ACC:{:.2f}, DBS:{:.2f}'.format(K, ACC, DBS))
    print('-(DBS + 1 * ACC): ', -(DBS + 1 * ACC))
    return -(DBS + 1 * ACC)  


def estimate_k_bybrent(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, phase, args):
    """
    :param labeled_feats:        labeled data features    n1*m
    :param labeled_targets:      labeled data labels      n1*1
    :param unlabeled_feats:      unlabeled data features  n2*m
    :param num_known_class:      number of known classes
    :param phase:                current phase
    :param args:                 various parameters
    :return:                     semi-supervised performance indicators average clustering accuracy (ACC) and Davies Bouldin score (DBS)
    """
    test_k_means_partial = partial(ss_kmeans_for_search, labeled_feats=labeled_feats, labeled_targets=labeled_targets, unlabeled_feats=unlabeled_feats, num_known_class=num_known_class, args=args)
    res = minimize_scalar(test_k_means_partial, bounds=(num_known_class+1, args.max_K), method='bounded', options={'disp': True}, tol=1) 
    best_k = int(res.x)
    print("The best K is {} for {}".format(best_k, phase))
    return best_k


# def estimate_k_byGA(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, phase, args):
#     test_k_means_partial = partial(ss_kmeans_for_search, labeled_feats=labeled_feats, labeled_targets=labeled_targets, unlabeled_feats=unlabeled_feats, num_known_class=num_known_class, args=args)
#     # ga = GA(func=test_k_means_partial, n_dim=1, size_pop=10, max_iter=3, lb=[num_known_class], ub=[args.max_K], precision=1)
#     ga = DE(func=test_k_means_partial, n_dim=1, size_pop=30, max_iter=1, lb=[num_known_class], ub=[args.max_K],) 
#     best_x, best_y = ga.run()
#     best_k = int(best_x[0])
#     print("The best K is {} for {}".format(best_k, phase))
#     return best_k


# def estimate_k_byGA(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, phase, args):
#     recode_scores = []
#     for k in np.arange(num_known_class, args.max_K):
#         recode_scores.append(ss_kmeans_for_search(k, labeled_feats=labeled_feats, labeled_targets=labeled_targets, unlabeled_feats=unlabeled_feats, num_known_class=num_known_class, args=args))
#
#     import matplotlib
#     import matplotlib.pyplot as plt
#     matplotlib.use('TkAgg')
#     plt.plot(np.arange(num_known_class, args.max_K), recode_scores, color="red")
#     plt.show()
#
#     best_k = np.arange(num_known_class, args.max_K)[np.argmax(np.array(recode_scores))]
#     print("The best K is {} for {}".format(best_k, phase))
#     return best_k
