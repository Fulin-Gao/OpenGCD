from methods.novel_category_discover.novel_category_discovery import gncd
from methods.open_set_recognition.osr_uncertainty import osr_uncertainty
from data.get_datasets import get_datasets, get_class_splits
from methods.exemplars_selection.es import sampling_byDS3
from sklearn.model_selection import train_test_split
from methods.closed_set_recognition.csr import csr
from project_utils.general_utils import seed_torch
from project_utils.cluster_utils import str2bool
from config import feature_extract_dir
from data.data_load import load_data
import numpy as np
import argparse
import math
import time
import os


if __name__ == "__main__":
    T1 = time.time()

    parser = argparse.ArgumentParser(
            description='OpenGCD_imagenet_100',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_K', default=500, type=int, help='Maximum number of categories acceptable')
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir, help='Feature storage address')
    parser.add_argument('--semi_sup', type=str2bool, default=True)
    parser.add_argument('--max_kmeans_iter', type=int, default=200)
    parser.add_argument('--k_means_init', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='imagenet_100', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.83, help='Percentage of training samples (5:1)')
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--class_splits', default=[40, 60, 80, 100], type=list, help='Split old and new classes')
    parser.add_argument('--alpha', type=list, default=[4, 1, 1], help='Adjusting for uncertainty')
    parser.add_argument('--reg', type=float, default=0.05, help='Penalty factor for DS3')
    parser.add_argument('--memory', type=int, default=10000, help='Buffer size')
    parser.add_argument('--classifier', type=str, default='head', help='options:{XGBoost, SVM, MLP}')

    # ----------------------
    # INIT
    # ----------------------
    print('Setting parameters...')
    args = parser.parse_args()
    seed_torch(0)
    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')

    # Specify known and unknown classes
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    print(args)

    # --------------------
    # DATA PREPARATION
    # --------------------
    print('Loading data...')
    train_transform, test_transform = None, None
    train_dataset, test_dataset, _, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)
    train_feats, test_feats, train_targets, test_targets = load_data(train_dataset, test_dataset, args)

    # --------------------
    # 1st PHASE
    # --------------------
    # Enter the initial stage and change the number of known classes
    cur_phase = '1st'
    num_known_class = args.class_splits[0]
    print('--'*5, 'Enter the {} stage, the number of currently known classes is {}'.format(cur_phase, num_known_class),'--'*5)
    train_feats_available = train_feats[train_targets < num_known_class]
    train_targets_available = train_targets[train_targets < num_known_class]
    test_feats_available = test_feats[test_targets < num_known_class]
    test_targets_available = test_targets[test_targets < num_known_class]

    # --------------------
    # 1st EXEMPLARS SELECTION
    # --------------------
    print('Performing {} exemplars selection...'.format(cur_phase))
    train_feats_exemplar, train_targets_exemplar = sampling_byDS3(train_feats_available, train_targets_available, num_known_class, args.reg,
                                                                  math.ceil(args.memory / num_known_class), cur_phase, args)

    # --------------------
    # 1st CLOSED-SET RECOGNITION
    # --------------------
    print('Performing {} closed-set recognition...'.format(cur_phase))
    model_csr1 = csr(train_feats_exemplar, test_feats_available, train_targets_exemplar, test_targets_available, num_known_class, cur_phase, args)

    # --------------------
    # 1st OPEN-SET RECOGNITION
    # --------------------
    print('Performing {} open-set recognition...'.format(cur_phase))
    num_unknown_class = args.class_splits[1] - num_known_class  # It is used only to fetch data and is unknown to the model
    _, predict_label_osr, HNA, OSFM, unknown_feats, online_feats_osr, online_targets_osr = osr_uncertainty(test_feats_available, test_targets_available, test_feats, test_targets, train_feats, train_targets, num_known_class, num_unknown_class, cur_phase, args, alpha=args.alpha[0],
                                                                     model=model_csr1)

    # --------------------
    # 1st GENERALIZE NOVEL CATEGORY DISCOVERY
    # --------------------
    print('Performing {} generalized novel category discovery...'.format(cur_phase))
    predict_label_ncd = gncd(train_feats_exemplar, train_targets_exemplar, unknown_feats, predict_label_osr, online_targets_osr, num_known_class, cur_phase, args)

    # --------------------
    # 1st MANUAL INTERVENTION
    # --------------------
    # Fetch new cluster data
    mask_novel = np.zeros(len(predict_label_ncd)) != 0
    mask_novel[len(test_targets_available):] = predict_label_ncd[len(test_targets_available):] >= 0  # num_known_class

    # Handle the novel cluster manually
    feats_novel = online_feats_osr[mask_novel]
    targets_novel = online_targets_osr[mask_novel]

    # Split the training and test sets
    train_feats_novel, test_feats_novel, train_targets_novel, test_targets_novel = train_test_split(feats_novel, targets_novel, random_state=104, test_size=1-args.prop_train_labels, shuffle=True)

    # Prepare currently available training sets and test sets
    train_feats_available = np.concatenate((train_feats_exemplar, train_feats_novel), axis=0)
    train_targets_available = np.concatenate((train_targets_exemplar, train_targets_novel))
    test_feats_available = np.concatenate((test_feats_available, test_feats_novel), axis=0)
    test_targets_available = np.concatenate((test_targets_available, test_targets_novel))

    # --------------------
    # 2nd PHASE
    # --------------------
    # Enter a new stage and change the number of known classes
    cur_phase = '2nd'
    num_known_class = args.class_splits[1]
    print('--'*5, 'Enter the {} stage, the number of currently known classes is {}'.format(cur_phase, num_known_class),'--'*5)

    # --------------------
    # 2nd EXEMPLARS SELECTION
    # --------------------
    print('Performing {} exemplars selection...'.format(cur_phase))
    train_feats_exemplar, train_targets_exemplar = sampling_byDS3(train_feats_available, train_targets_available, num_known_class, args.reg,
                                                                  math.ceil(args.memory / num_known_class), cur_phase, args)

    # --------------------
    # 2nd CLOSED-SET RECOGNITION
    # --------------------
    print('Performing {} closed-set recognition...'.format(cur_phase))
    model_csr2 = csr(train_feats_exemplar, test_feats_available, train_targets_exemplar, test_targets_available, num_known_class, cur_phase, args)

    # --------------------
    # 2nd OPEN-SET RECOGNITION
    # --------------------
    print('Performing {} open-set recognition...'.format(cur_phase))
    num_unknown_class = args.class_splits[2] - num_known_class  # It is used only to fetch data and is unknown to the model
    _, predict_label_osr, HNA, OSFM, unknown_feats, online_feats_osr, online_targets_osr = osr_uncertainty(test_feats_available, test_targets_available, test_feats, test_targets, train_feats, train_targets, num_known_class, num_unknown_class, cur_phase, args, alpha=args.alpha[1],
                                                                     model=model_csr2)

    # --------------------
    # 2nd GENERALIZE NOVEL CATEGORY DISCOVERY
    # --------------------
    print('Performing {} generalized novel category discovery...'.format(cur_phase))
    predict_label_ncd = gncd(train_feats_exemplar, train_targets_exemplar, unknown_feats, predict_label_osr, online_targets_osr, num_known_class, cur_phase, args)

    # --------------------
    # 2nd MANUAL INTERVENTION
    # --------------------
    # Fetch new cluster data
    mask_novel = np.zeros(len(predict_label_ncd)) != 0
    mask_novel[len(test_targets_available):] = predict_label_ncd[len(test_targets_available):] >= 0  # num_known_class

    # Handle the novel cluster manually
    feats_novel = online_feats_osr[mask_novel]
    targets_novel = online_targets_osr[mask_novel]

    # Split the training and test sets
    train_feats_novel, test_feats_novel, train_targets_novel, test_targets_novel = train_test_split(feats_novel,
                                                                                                    targets_novel,
                                                                                                    random_state=104,
                                                                                                    test_size=1 - args.prop_train_labels,
                                                                                                    shuffle=True)

    # Prepare currently available training sets and test sets
    train_feats_available = np.concatenate((train_feats_exemplar, train_feats_novel), axis=0)
    train_targets_available = np.concatenate((train_targets_exemplar, train_targets_novel))
    test_feats_available = np.concatenate((test_feats_available, test_feats_novel), axis=0)
    test_targets_available = np.concatenate((test_targets_available, test_targets_novel))

    # --------------------
    # 3rd PHASE
    # --------------------
    # Enter a new stage and change the number of known classes
    cur_phase = '3rd'
    num_known_class = args.class_splits[2]
    print('--' * 5,
          'Enter the {} stage, the number of currently known classes is {}'.format(cur_phase, num_known_class),
          '--' * 5)

    # --------------------
    # 3rd EXEMPLARS SELECTION
    # --------------------
    print('Performing {} exemplars selection...'.format(cur_phase))
    train_feats_exemplar, train_targets_exemplar = sampling_byDS3(train_feats_available, train_targets_available, num_known_class, args.reg,
                                                                  math.ceil(args.memory / num_known_class), cur_phase, args)

    # --------------------
    # 3rd CLOSED-SET RECOGNITION
    # --------------------
    print('Performing {} closed-set recognition...'.format(cur_phase))
    model_csr3 = csr(train_feats_exemplar, test_feats_available, train_targets_exemplar, test_targets_available, num_known_class, cur_phase, args)

    # --------------------
    # 3rd OPEN-SET RECOGNITION
    # --------------------
    print('Performing {} open-set recognition...'.format(cur_phase))
    num_unknown_class = args.class_splits[3] - num_known_class  # It is used only to fetch data and is unknown to the model
    _, predict_label_osr, HNA, OSFM, unknown_feats, online_feats_osr, online_targets_osr = osr_uncertainty(test_feats_available, test_targets_available, test_feats, test_targets, train_feats, train_targets, num_known_class, num_unknown_class, cur_phase, args, alpha=args.alpha[2],
                                                                     model=model_csr3)

    # --------------------
    # 3rd GENERALIZE NOVEL CATEGORY DISCOVERY
    # --------------------
    print('Performing {} generalized novel category discovery...'.format(cur_phase))
    predict_label_ncd = gncd(train_feats_exemplar, train_targets_exemplar, unknown_feats, predict_label_osr, online_targets_osr, num_known_class, cur_phase, args)

    # --------------------
    # 3rd MANUAL INTERVENTION
    # --------------------
    # Fetch new cluster data
    mask_novel = np.zeros(len(predict_label_ncd)) != 0
    mask_novel[len(test_targets_available):] = predict_label_ncd[len(test_targets_available):] >= 0 # num_known_class

    # Handle the novel cluster manually
    feats_novel = online_feats_osr[mask_novel]
    targets_novel = online_targets_osr[mask_novel]

    # Split the training and test sets
    train_feats_novel, test_feats_novel, train_targets_novel, test_targets_novel = train_test_split(feats_novel,
                                                                                                    targets_novel,
                                                                                                    random_state=104,
                                                                                                    test_size=1 - args.prop_train_labels,
                                                                                                    shuffle=True)

    # Prepare currently available training sets and test sets
    train_feats_available = np.concatenate((train_feats_exemplar, train_feats_novel), axis=0)
    train_targets_available = np.concatenate((train_targets_exemplar, train_targets_novel))
    test_feats_available = np.concatenate((test_feats_available, test_feats_novel), axis=0)
    test_targets_available = np.concatenate((test_targets_available, test_targets_novel))

    # --------------------
    # 4th PHASE
    # --------------------
    # Enter a new stage and change the number of known classes
    cur_phase = '4th'
    num_known_class = args.class_splits[3]
    print('--' * 5,
          'Enter the {} stage, the number of currently known classes is {}'.format(cur_phase, num_known_class),
          '--' * 5)

    # --------------------
    # 4th EXEMPLARS SELECTION
    # --------------------
    print('Performing {} exemplars selection...'.format(cur_phase))
    # Making exemplars selections
    train_feats_exemplar, train_targets_exemplar = sampling_byDS3(train_feats_available, train_targets_available, num_known_class, args.reg,
                                                                  math.ceil(args.memory / num_known_class), cur_phase, args)

    # --------------------
    # 4th CLOSED-SET RECOGNITION
    # --------------------
    print('Performing {} closed-set recognition...'.format(cur_phase))
    model_csr3 = csr(train_feats_exemplar, test_feats_available, train_targets_exemplar, test_targets_available, num_known_class, cur_phase, args)

    T2 = time.time()
    print('程序运行时间:{}秒'.format(T2 - T1))
    print('Done!')

