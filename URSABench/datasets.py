import os

import numpy as np
import torch
import torchvision
from torch.utils.data import random_split

from URSABench import util
from URSABench.util import DatasetFromSubset

c10_classes = np.array([
    [0, 1, 2, 8, 9],
    [3, 4, 5, 6, 7]
], dtype=np.int32)


def camvid_loaders(path, batch_size, num_workers, transform_train, transform_test,
                   use_validation, val_size, shuffle_train=True,
                   joint_transform=None, ft_joint_transform=None, ft_batch_size=1, **kwargs):
    # load training and finetuning datasets
    print(path)
    train_set = CamVid(root=path, split='train', joint_transform=joint_transform, transform=transform_train, **kwargs)
    ft_train_set = CamVid(root=path, split='train', joint_transform=ft_joint_transform, transform=transform_train,
                          **kwargs)

    val_set = CamVid(root=path, split='val', joint_transform=None, transform=transform_test, **kwargs)
    test_set = CamVid(root=path, split='test', joint_transform=None, transform=transform_test, **kwargs)

    num_classes = 11  # hard coded labels ehre

    return {'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    ),
               'fine_tune': torch.utils.data.DataLoader(
                   ft_train_set,
                   batch_size=ft_batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'val': torch.utils.data.DataLoader(
                   val_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               )}, num_classes


def svhn_loaders(path, batch_size, num_workers, transform_train, transform_test, use_validation, val_size,
                 shuffle_train=True):
    train_set = torchvision.datasets.SVHN(root=path, split='train', download=True, transform=transform_train)

    if use_validation:
        test_set = torchvision.datasets.SVHN(root=path, split='train', download=True, transform=transform_test)
        train_set.data = train_set.data[:-val_size]
        train_set.labels = train_set.labels[:-val_size]

        test_set.data = test_set.data[-val_size:]
        test_set.labels = test_set.labels[-val_size:]

    else:
        print('You are going to run models on the test set. Are you sure?')
        test_set = torchvision.datasets.SVHN(root=path, split='test', download=True, transform=transform_test)
        test_set.data = test_set.data[:10000]
        test_set.labels = test_set.labels[:10000]
        print('SVHN: ',len(test_set.labels))
    num_classes = 10

    return \
        {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes


def tin_loaders(path, batch_size, num_workers, transform_train, transform_test,
                use_validation, val_size, shuffle_train=True,
                joint_transform=None, ft_joint_transform=None, ft_batch_size=1, **kwargs):
    train_path = path + '/train/'
    test_path = path + '/test/'
    num_classes = 200
    if use_validation:
        train_set = torchvision.datasets.ImageFolder(root=train_path)
        lengths = [int(len(train_set) * (1 - val_size)), int(len(train_set) * val_size)]
        train_subset, val_subset = random_split(train_set, lengths)
        train_set = DatasetFromSubset(train_subset, transform=transform_train)
        test_set = DatasetFromSubset(val_subset, transform=transform_test)
        print("Using train (" + str(len(train_set)) + ") + validation (" + str(len(test_set)) + ")")
    else:
        train_set = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
        test_set = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)

    return \
        {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes


def loaders(dataset, path, batch_size, num_workers, transform_train, transform_test,
            use_validation=True, val_size=0.2, split_classes=None, shuffle_train=True,
            imbalance = False, **kwargs):
    if dataset == 'CamVid':
        return camvid_loaders(path, batch_size=batch_size, num_workers=num_workers, transform_train=transform_train,
                              transform_test=transform_test, use_validation=use_validation, val_size=val_size, **kwargs)
    if dataset == 'TIN':
        return tin_loaders(path, batch_size=batch_size, num_workers=num_workers, transform_train=transform_train,
                           transform_test=transform_test, use_validation=use_validation, val_size=val_size, **kwargs)

    if dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'MNIST':
        path = os.path.join(path, dataset.lower())
        ds = getattr(torchvision.datasets, dataset)

    ds = getattr(torchvision.datasets, dataset)

    if dataset == 'SVHN':
        return svhn_loaders(path, batch_size, num_workers, transform_train, transform_test, use_validation, val_size)
    else:

        ds = getattr(torchvision.datasets, dataset)

    if dataset == 'STL10':
        train_set = ds(root=path, split='train', download=True, transform=transform_train)
        num_classes = 10
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9])
        train_set.labels = cls_mapping[train_set.labels]
    elif dataset == 'LSUN' or dataset == 'CelebA':
        train_set = ds(path, 'train', transform=transform_train, download=True)
        # import pdb; pdb.set_trace()
        num_classes = max(train_set.targets) + 1
    else:
        train_set = ds(root=path, train=True, download=True, transform=transform_train)
        num_classes = max(train_set.targets) + 1

    if dataset == 'MNIST' and imbalance == True:
        labels = [3,7]
        print('Decision Task MNIST')
        for l in labels:
            new_labels, new_data = util.increase_data_imbalance(label=l, dataset=train_set, remove_frac = 0.99)
            train_set.data = new_data
            train_set.targets = new_labels

    if dataset == 'CIFAR10' and imbalance == True:
        labels = [0,1,8,9]
        print('Decision Task CIFAR10')
        for l in labels:
            new_labels, new_data = util.increase_data_imbalance(label=l, dataset=train_set, remove_frac = 0.9)
            train_set.data = new_data
            train_set.targets = new_labels

    if dataset == 'CIFAR100' and imbalance == True:
        labels = [58, 69, 85]
        print('Decision Task CIFAR100')
        for l in labels:
            new_labels, new_data = util.increase_data_imbalance(label=l, dataset=train_set, remove_frac = 0.9)
            train_set.data = new_data
            train_set.targets = new_labels


    if use_validation:
        val_size = int(len(train_set.data) * val_size)
        r_ind = torch.randperm(len(train_set.data))

        print("Using train (" + str(len(train_set.data) - val_size) + ") + validation (" + str(val_size) + ")")
        train_set.data = train_set.data[r_ind[:-val_size]]
        train_set.targets = torch.LongTensor(train_set.targets)
        train_set.targets = train_set.targets[r_ind[:-val_size]]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.data = test_set.data[r_ind[-val_size:]]
        test_set.targets = torch.LongTensor(test_set.targets)
        test_set.targets = test_set.targets[r_ind[-val_size:]]

    else:
        print('You are going to run models on the test set. Are you sure?')
        if dataset == 'STL10':
            test_set = ds(root=path, split='test', download=True, transform=transform_test)
            test_set.labels = cls_mapping[test_set.labels]
            print('STL10: ',len(test_set.labels))
        elif dataset == 'LSUN' or dataset == 'CelebA':
            train_set = ds(path, 'test', transform=transform_train, download=True)
        else:
            test_set = ds(root=path, train=False, download=True, transform=transform_test)

    if split_classes is not None:
        assert dataset == 'CIFAR10'
        assert split_classes in {0, 1}

        print('Using classes:', end='')
        print(c10_classes[split_classes])
        train_mask = np.isin(train_set.targets, c10_classes[split_classes])
        train_set.data = train_set.data[train_mask, :]
        train_set.targets = np.array(train_set.targets)[train_mask]
        train_set.targets = np.where(train_set.targets[:, None] == c10_classes[split_classes][None, :])[
            1].tolist()
        print('Train: %d/%d' % (train_set.data.shape[0], train_mask.size))

        test_mask = np.isin(test_set.targets, c10_classes[split_classes])
        test_set.data = test_set.data[test_mask, :]
        test_set.targets = np.array(test_set.targets)[test_mask]
        test_set.targets = np.where(test_set.targets[:, None] == c10_classes[split_classes][None, :])[
            1].tolist()
        print('Test: %d/%d' % (test_set.data.shape[0], test_mask.size))

    return \
        {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes


def loaders_inc(dataset, path, num_chunks, batch_size, num_workers, transform_train, transform_test,
                use_validation=True, val_size=5000, shuffle_train=True, seed=1):
    assert dataset in {'MNIST', 'CIFAR10', 'CIFAR100'}
    path = os.path.join(path, dataset.lower())

    ds = getattr(torchvision.datasets, dataset)

    train_set = ds(root=path, train=True, download=True, transform=transform_train)
    num_classes = int(max(train_set.targets)) + 1

    num_samples = (train_set.data.shape[0] - val_size) if use_validation else train_set.data.shape[0]
    train_sets = list()
    offset = 0

    random_state = np.random.RandomState(seed)
    order = random_state.permutation(train_set.data.shape[0])

    for i in range(num_chunks, 0, -1):
        chunk_size = (num_samples + i - 1) // i
        tmp_set = ds(root=path, train=True, download=True, transform=transform_train)
        tmp_set.data = tmp_set.data[order[offset:offset + chunk_size]]
        tmp_set.targets = np.array(tmp_set.targets)[order[offset:offset + chunk_size]]

        train_sets.append(tmp_set)
        offset += chunk_size
        num_samples -= chunk_size

    print('Using train %d chunks: %s' % (num_chunks, str([tmp_set.data.shape[0] for tmp_set in train_sets])))

    if use_validation:
        print('Using validation (%d)' % val_size)

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.data = test_set.data[order[-val_size:]]
        test_set.targets = np.array(test_set.targets)[order[-val_size:]]
        delattr(test_set, 'train_data')
        delattr(test_set, 'train_labels')
    else:
        print('You are going to run models on the test set. Are you sure?')

        test_set = ds(root=path, train=False, download=True, transform=transform_test)

    return \
        {
            'train': [
                torch.utils.data.DataLoader(
                    tmp_set,
                    batch_size=batch_size,
                    shuffle=True and shuffle_train,
                    num_workers=num_workers,
                    pin_memory=True
                ) for tmp_set in train_sets
            ],
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes
