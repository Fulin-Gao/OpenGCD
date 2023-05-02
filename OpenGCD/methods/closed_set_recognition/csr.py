from project_utils.metrics import evaluation_closed
from models.XGBoost import xgboost
from models.SVM import svm
from models.MLP import mlp


def csr(train_feats, test_feats, train_targets, test_targets, num_known_class, phase, args):
    """
    :param train_feats:       training features n1*m1
    :param test_feats:        test features n2*m2
    :param train_targets:     training labels n1*1
    :param test_targets:      test labels n2*1
    :param num_known_class:   number of training set classes
    :param phase:             current stage
    :param args:              various parameters
    :return:                  return model for open-set recognition
    """
    # Prepare available training features and labels, test features and labels
    train_feats_csr = train_feats[train_targets < num_known_class]
    test_feats_csr = test_feats[test_targets < num_known_class]
    train_targets_csr = train_targets[train_targets < num_known_class]
    test_targets_csr = test_targets[test_targets < num_known_class]

    # Classifier fitting, prediction
    if args.classifier == 'XGBoost':
        predict_prob_csr, predict_label_csr, model_csr = xgboost(train_feats_csr, train_targets_csr, test_feats_csr, test_targets_csr, num_known_class)
    elif args.classifier == 'SVM':
        predict_prob_csr, predict_label_csr, model_csr = svm(train_feats_csr, train_targets_csr, test_feats_csr)
    elif args.classifier == 'MLP':
        predict_prob_csr, predict_label_csr, model_csr = mlp(train_feats_csr, train_targets_csr, test_feats_csr)
    else:
        raise NotImplementedError

    # Evaluate closed-set results
    evaluation_closed(test_targets_csr, predict_label_csr, phase, args)

    return model_csr
