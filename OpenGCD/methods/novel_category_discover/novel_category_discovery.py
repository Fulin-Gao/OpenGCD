from methods.novel_category_discover.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from methods.novel_category_discover.estimate_k import estimate_k_bybrent
from methods.novel_category_discover.estimate_k import estimate_k_byGA
from methods.novel_category_discover.estimate_k import estimate_k
from project_utils.metrics import evaluation_novel
import torch
import copy


def gncd(train_feats_exemplar, train_targets_exemplar, unknown_feats, predict_label_osr, online_targets_osr, num_known_class, phase, args):
    """
    :param train_feats_exemplar:      labeled exemplar features                 n1*m
    :param train_targets_exemplar:    labeled exemplar labels                   n1*1
    :param unknown_feats:             isolated unknown features (unlabeled)     n2*m
    :param predict_label_osr:         labels predicted by the open-set in the previous step  n3*1
    :param online_targets_osr:        ground true labels of online data         n3*1
    :param num_known_class:           number of known classes
    :param phase:                     current stage
    :param args:                      various parameters
    :return:                          generalized novel category discover results
    """
    # Prepare known labeled data and unknown unlabeled data
    labeled_feats, labeled_targets = train_feats_exemplar, train_targets_exemplar
    unlabeled_feats = unknown_feats

    # Estimate total number of classes K
    print('Estimating the total number of classes K')
    best_K = estimate_k_bybrent(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, phase, args)  # estimate_k_bybrent

    # Perform semi-supervised k-means clustering based on the best K
    print('Performing semi-supervised k-means by best K={}'.format(best_K))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    l_feats, u_feats, l_targets = (torch.from_numpy(x).to(device) for x in
                                              (labeled_feats, unlabeled_feats, labeled_targets))
    ss_k_means = SemiSupKMeans(k=best_K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                               mode=None)
    ss_k_means.fit_mix(u_feats, l_feats, l_targets)

    # Get predictions for unknown unlabeled data
    all_pred = ss_k_means.labels_.cpu().numpy()
    unlabeled_pred = all_pred[len(labeled_targets):]

    # Adjust prediction labels for unknown unlabeled data based on semi-supervised clustering results
    # Those considered as error isolated will be put back to the known class, and those still considered as unknown will be assigned to the corresponding cluster
    predict_label_osr[predict_label_osr == num_known_class] = unlabeled_pred

    # Evaluate generalized novel category discovery results
    #
    evaluation_novel(online_targets_osr, predict_label_osr, phase, args)

    return predict_label_osr
