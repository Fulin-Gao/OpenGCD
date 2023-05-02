from project_utils.metrics import evaluation_open
from models.XGBoost import xgboost
from models.SVM import svm
from models.MLP import mlp
import numpy as np


def osr_uncertainty(test_feats_csr, test_targets_csr, test_feats, test_targets, train_feats, train_targets, num_known_class, num_unknown_class, phase, args, alpha=1, model=None):
    """
    :param test_feats_csr:        known class test features     n1*m
    :param test_targets_csr:      known class test labels       n1*1
    :param test_feats:            all test features             n2*m
    :param test_targets:          all test labels               n2*1
    :param train_feats:           all train features  n2*m      n3*m
    :param train_targets:         all train labels    n2*1      n3*1
    :param num_known_class:       number of known classes
    :param num_unknown_class:     number of Unknown classes, here it is just to take the data, the online open-set recognition process is unknown by default for this parameter.
    :param phase:                 current stage
    :param alpha:                 a factor that regulates uncertainty
    :param model:                 the model fitted by the previous closed-set recognition
    :return:                      prediction for online data (probability and label), open-set prediction performance scores (HNA, OSFM), isolate data features, features for online data, ground true labels for online data
    """
    # Prepare features and labels (Including known in the test set and unknown classes in training set and test set)
    train_feats_osr = train_feats[
        (num_known_class <= train_targets) & (train_targets < num_known_class + num_unknown_class)]
    test_feats_osr = test_feats[
        (num_known_class <= test_targets) & (test_targets < num_known_class + num_unknown_class)]
    online_feats_osr = np.concatenate((test_feats_csr, train_feats_osr, test_feats_osr), axis=0)

    train_targets_osr = train_targets[
        (num_known_class <= train_targets) & (train_targets < num_known_class + num_unknown_class)]
    test_targets_osr = test_targets[
        (num_known_class <= test_targets) & (test_targets < num_known_class + num_unknown_class)]
    online_targets_osr1 = np.concatenate((test_targets_csr, train_targets_osr, test_targets_osr))  # ground true labels for novel category discovery
    online_targets_osr = np.concatenate((test_targets_csr, num_known_class * np.ones(len(train_targets_osr) + len(test_targets_osr), dtype=int)))  # unknown class is set as num_known_class-th class

    # Perform closed-set recognition through previous model
    if args.classifier == 'XGBoost':
        predict_prob_csr, predict_label_csr, model_csr = xgboost(None, None, online_feats_osr, online_targets_osr, num_known_class, model=model)
    elif args.classifier == 'SVM':
        predict_prob_csr, predict_label_csr, model_csr = svm(None, None, online_feats_osr, model=model)
    elif args.classifier == 'MLP':
        predict_prob_csr, predict_label_csr, model_csr = mlp(None, None, online_feats_osr, model=model)
    else:
        raise NotImplementedError

    # Calculate uncertainty
    num_sample = len(predict_label_csr)
    mask = np.ones((num_sample, num_known_class))
    for idx, p_l in enumerate(predict_label_csr):
        mask[idx, p_l] = 0
    mask = mask.astype(bool)
    uncertainty = alpha * np.sum(predict_prob_csr[mask].reshape(num_sample, num_known_class - 1), axis=1)

    # Get open-set prediction
    prob_ass = np.concatenate((predict_prob_csr, uncertainty.reshape(-1, 1)), axis=1)
    label_ass = np.argmax(prob_ass, axis=1)

    # Evaluate open-set results
    HNA, OSFM = evaluation_open(online_targets_osr, label_ass, phase)

    # Isolate data belonging to unknown classes based on the open-set results
    unknown_feats = online_feats_osr[label_ass == num_known_class]

    return prob_ass, label_ass, HNA, OSFM, unknown_feats, online_feats_osr, online_targets_osr1
