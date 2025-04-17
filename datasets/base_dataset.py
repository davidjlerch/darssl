# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import copy
# import mmcv
import numpy as np
import os.path as osp
import torch
import pickle
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
# from mmcv.utils import print_log
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
import numpy as np

from common import utils


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.None

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.
    - Methods:`prepare_train_frames`, providing train data.
    - Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (callable): A sequence of data transforms.
        data_prefix (str): Path to a directory where videos are held. Default: ''.
        test_mode (bool): Store True when building test or validation dataset. Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class dataset. Default: False.
        num_classes (int | None): Number of classes of the dataset, used in multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of different filename format. However,
            if taking videos as input, it should be set to 0, since frames loaded from videos count from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'. Default: 'RGB'.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 frame_masker=None,
                 joint_masker=None,
                 noiser=None,
                 augmentation=None,
                 data_prefix='',
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 length=None,
                 start_index=1):
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.length = length

        self.pipeline = pipeline
        self.frame_masker = frame_masker
        self.joint_masker = joint_masker
        self.noiser = noiser
        self.augmentation = augmentation
        self.video_infos = self.load_annotations()
        self.mask_pad_value = 1

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        # video_infos = mmcv.load(self.ann_file)
        video_infos = pickle.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return pickle.dump(results, out)

    def select_body(self, body_data):
        if body_data.shape[0] > 1:
            motion = np.var(body_data, axis=tuple(range(1, body_data.ndim)), dtype=np.float32)
            # We do not convert numpy to tensor here,
            # unless we change the augmentation to support torch tensor
            # return torch.from_numpy(body_data[motion.argmax()]).type(torch.float32)
            return body_data[motion.argmax()]
        # return torch.from_numpy(body_data[0]).type(torch.float32)
        return body_data[0]

    def prepare_frames(self, idx):
        """Prepare the frames for training given the index."""
        # Note Zeyun: did not find any difference between prepare_train_frames
        # and prepare_test_frames, so we keep only one method
        results = copy.deepcopy(self.video_infos[idx])
        results['start_index'] = self.start_index
        results['keypoint'] = self.select_body(results['keypoint'])

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        # if self.multi_class and isinstance(results['label'], list):
        onehot = torch.zeros(self.num_classes)
        onehot[results['label']] = 1.
        results['label'] = onehot

        if self.pipeline:
            self.pipeline(results)

            if 'body_center' not in results:
                print(self.pipeline, results.keys())

        results['input'] = copy.deepcopy(results['keypoint'])
        results['frame_mask'] = copy.deepcopy(results['keypoint'])
        results['joint_mask'] = copy.deepcopy(results['keypoint'])
        results['noise'] = copy.deepcopy(results['keypoint'])

        frame_mask = None
        joint_mask = None

        if self.frame_masker:
            self.frame_masker(results)
            results['frame_mask'] = results['input']
            results['input'] = copy.deepcopy(results['keypoint'])
            if self.length is not None:
                full_arr = np.zeros((self.length, results['frame_mask'].shape[1], results['frame_mask'].shape[2]))
                full_arr[:len(results['frame_mask'])] = results['frame_mask']
                results['frame_mask'] = copy.deepcopy(full_arr)
            results['frame_mask'] = torch.from_numpy(results['frame_mask'].astype(np.float32))

        if self.joint_masker:
            self.joint_masker(results)
            results['joint_mask'] = results['input']
            results['input'] = copy.deepcopy(results['keypoint'])
            if self.length is not None:
                full_arr = np.zeros((self.length, results['joint_mask'].shape[1], results['joint_mask'].shape[2]))
                full_arr[:len(results['joint_mask'])] = results['joint_mask']
                results['joint_mask'] = copy.deepcopy(full_arr)
            results['joint_mask'] = torch.from_numpy(results['joint_mask'].astype(np.float32))

        if self.augmentation:
            self.augmentation(results)
            results['augment'] = results['input']
            if self.length is not None:
                full_arr = np.zeros((self.length, results['augment'].shape[1], results['augment'].shape[2]))
                full_arr[:len(results['augment'])] = results['augment']
                results['augment'] = copy.deepcopy(full_arr)
            results['augment'] = torch.from_numpy(results['augment'].astype(np.float32))

        if self.noiser:
            self.noiser(results)
            results['noise'] = results['input']
            if joint_mask is not None:
                results['input'] = results['input'] * joint_mask
            if frame_mask is not None:
                results['input'] = results['input'] * frame_mask
            if self.length is not None:
                full_arr = np.zeros((self.length, results['noise'].shape[1], results['noise'].shape[2]))
                full_arr[:len(results['noise'])] = results['noise']
                results['noise'] = copy.deepcopy(full_arr)
            results['noise'] = torch.from_numpy(results['noise'].astype(np.float32))

        if self.augmentation:
            self.augmentation(results)

        results['keypoint'] = torch.from_numpy(results['keypoint'].astype(np.float32))
        results['input'] = torch.from_numpy(results['input'].astype(np.float32))
        return results

    def collate_fn(self, batch):
        """custom collate function for dealing with samples with different sequence length"""
        pose_key_input = 'input'
        pose_key_fm = 'frame_mask'
        pose_key_jm = 'joint_mask'
        pose_key_noisy = 'noise'
        pose_key_orig = 'keypoint'
        pose_key_aug = 'augment'

        total_frames = torch.Tensor([item[pose_key_orig].shape[0] for item in batch]).type(torch.int64)
        b_pose_input = [item.pop(pose_key_input) for item in batch]
        b_pose_input, mask = utils.pad_sequence(
            b_pose_input, padding_value=0, mask_padding_value=self.mask_pad_value,
            return_mask=True,
        )
        b_pose_fm = [item.pop(pose_key_fm) for item in batch]
        b_pose_fm, mask = utils.pad_sequence(
            b_pose_fm, padding_value=0, mask_padding_value=self.mask_pad_value,
            return_mask=True,
        )
        b_pose_jm = [item.pop(pose_key_jm) for item in batch]
        b_pose_jm, mask = utils.pad_sequence(
            b_pose_jm, padding_value=0, mask_padding_value=self.mask_pad_value,
            return_mask=True,
        )
        b_pose_noisy = [item.pop(pose_key_noisy) for item in batch]
        b_pose_noisy, mask = utils.pad_sequence(
            b_pose_noisy, padding_value=0, mask_padding_value=self.mask_pad_value,
            return_mask=True,
        )
        b_pose_orig = [item.pop(pose_key_orig) for item in batch]
        b_pose_orig, mask_orig = utils.pad_sequence(
            b_pose_orig, padding_value=0, mask_padding_value=self.mask_pad_value,
            return_mask=True,
        )
        b_pose_aug = [item.pop(pose_key_aug) for item in batch]
        b_pose_aug, mask = utils.pad_sequence(
            b_pose_aug, padding_value=0, mask_padding_value=self.mask_pad_value,
            return_mask=True,
        )

        # assert torch.equal(mask, mask_orig)

        batch = default_collate(batch)
        batch.update({
            pose_key_input: b_pose_input.flatten(start_dim=2),
            pose_key_fm: b_pose_fm.flatten(start_dim=2),
            pose_key_jm: b_pose_jm.flatten(start_dim=2),
            pose_key_noisy: b_pose_noisy.flatten(start_dim=2),
            pose_key_orig: b_pose_orig.flatten(start_dim=2),
            pose_key_aug: b_pose_aug.flatten(start_dim=2),
            'mask': mask,
            'total_frames': total_frames
        })

        return batch

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        return self.prepare_frames(idx)


def save_scatter_plot_with_names(i, x, y):
    plt.scatter(x, y)

    # Add names to each point based on their index
    for index, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(index), fontsize=8, ha='right', va='bottom')

    plt.title(f'Scatter Plot {i}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)

    # Save the scatter plot as an image
    plt.savefig(f'scatter_plot_with_names_{i}.png')

    # Clear the current plot for the next iteration
    plt.clf()
