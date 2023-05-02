from project_utils.feature_vector_dataset import FeatureVectorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import os

def load_data(train_dataset, test_dataset, args):
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    # Convert to feature vector dataset
    train_dataset = FeatureVectorDataset(base_dataset=train_dataset, feature_root=os.path.join(args.save_dir, 'train'))
    train_dataset.target_transform = target_transform
    test_dataset = FeatureVectorDataset(base_dataset=test_dataset, feature_root=os.path.join(args.save_dir, 'test'))

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=False)

    # Extract all train features
    train_feats = []
    train_targets = np.array([])
    for batch_idx, (feats, label, _, _) in enumerate(tqdm(train_loader)):
        feats = torch.nn.functional.normalize(feats, dim=-1)
        train_feats.append(feats.cpu().numpy())
        train_targets = np.append(train_targets, label.cpu().numpy())
    train_feats = np.concatenate(train_feats)

    # Extract all test features
    test_feats = []
    test_targets = np.array([])
    for batch_idx, (feats, label, _) in enumerate(tqdm(test_loader)):
        feats = torch.nn.functional.normalize(feats, dim=-1)
        test_feats.append(feats.cpu().numpy())
        test_targets = np.append(test_targets, label.cpu().numpy())
    test_feats = np.concatenate(test_feats)

    return train_feats, test_feats, train_targets, test_targets