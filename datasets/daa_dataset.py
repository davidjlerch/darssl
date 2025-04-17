# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import numpy as np
from datasets.base_dataset import BaseDataset
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

def joint_smoothing_np(P):
    T, V, C = P.shape
    new_P = P.copy()  # Create a new skeleton sequence to avoid modifying the original

    for t in range(T):
        if T > t > 0 and np.sum(P[(t + 1) % T]) != 0 and np.sum(P[t - 1]) != 0:
            for v in range(V):
                if np.sum(P[t - 1][v]) != 0 and np.sum(P[(t + 1) % T][v]) != 0:
                    # If the next and previous frames for joint v are not missing
                    new_P[t][v] = (P[t - 1][v] + P[(t + 1) % T][v]) / 2
                elif t > 0 and np.sum(P[t - 1]) != 0:
                    new_P[t][v] = P[t - 1][v]
                elif np.sum(P[(t + 1) % T]) != 0:
                    new_P[t][v] = P[(t + 1) % T][v]
        elif t > 0 and np.sum(P[t - 1]) != 0:
            new_P[t] = P[t - 1]
        elif np.sum(P[(t + 1) % T]) != 0:
            new_P[t] = P[(t + 1) % T]
    return new_P


def joint_filling_np(P):
    # Assuming P is a np tensor with dimensions (T, V, C)
    T, V, C = P.shape
    new_P = P.copy()  # Create a new skeleton sequence to avoid modifying the original

    # Step 1: Process of Missing Frames
    for t in range(T):
        if np.sum(P[t]) == 0:  # If the sum of all joint coordinates at frame t is 0
            if np.sum(P) == 0:
                continue  # All frames are missing, can't perform smoothing
            # Circular Filling
            if T > t > 0 and np.sum(P[(t + 1) % T]) != 0 and np.sum(P[t - 1]) != 0:
                for v in range(V):
                    if np.sum(P[t - 1][v]) != 0 and np.sum(P[(t + 1) % T][v]) != 0:
                        # If the next and previous frames for joint v are not missing
                        new_P[t][v] = (P[t - 1][v] + P[(t + 1) % T][v]) / 2
                    elif t > 0 and np.sum(P[t - 1]) != 0:
                        new_P[t][v] = P[t - 1][v]
                    elif np.sum(P[(t + 1) % T]) != 0:
                        new_P[t][v] = P[(t + 1) % T][v]
            elif t > 0 and np.sum(P[t - 1]) != 0:
                new_P[t] = P[t - 1]
            elif np.sum(P[(t + 1) % T]) != 0:
                new_P[t] = P[(t + 1) % T]
    # Step 2: Process of Missing Joints
    for t in range(T):
        for v in range(V):
            if np.sum(P[t][v]) == 0:  # If the sum of coordinates for joint v at frame t is 0
                if T-1 > t > 0 and np.sum(P[t - 1][v]) != 0 and np.sum(P[(t + 1) % T][v]) != 0:
                    new_P[t][v] = (P[t - 1][v] + P[(t + 1) % T][v]) / 2
                elif t > 0 and np.sum(P[t - 1][v]) != 0:
                    new_P[t][v] = P[t - 1][v]
                elif np.sum(P[(t + 1) % T][v]) != 0:
                    new_P[t][v] = P[(t + 1) % T][v]

    return new_P


class DAADataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. For UCF101 and HMDB51, allowed choices are 'train1', 'test1',
            'train2', 'test2', 'train3', 'test3'. For NTURGB+D, allowed choices are 'xsub_train', 'xsub_val',
            'xview_train', 'xview_val'. For NTURGB+D 120, allowed choices are 'xsub_train', 'xsub_val', 'xset_train',
            'xset_val'. For FineGYM, allowed choices are 'train', 'val'. Default: None.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose. For a video with n frames, it is a
            valid training sample only if n * valid_ratio frames have human pose. None means not applicable (only
            applicable to Kinetics Pose). Default: None.
        box_thr (float): The threshold for human proposals. Only boxes with confidence score larger than `box_thr` is
            kept. None means not applicable (only applicable to Kinetics). Allowed choices are 0.5, 0.6, 0.7, 0.8, 0.9.
            Default: 0.5.
        class_prob (list | None): The class-specific multiplier, which should be a list of length 'num_classes', each
            element >= 1. The goal is to resample some rare classes to improve the overall performance. None means no
            resampling performed. Default: None.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
        **kwargs: Keyword arguments for 'BaseDataset'.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 frame_masker=None,
                 joint_masker=None,
                 noiser=None,
                 split=None,
                 part=None,
                 length=None,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 **kwargs):
        self.split = split
        self.part = part

        super().__init__(
            ann_file, pipeline, frame_masker, joint_masker, noiser, start_index=0, length=length, **kwargs)
        self.length = length
        # box_thr, which should be a string
        self.box_thr = box_thr
        self.class_prob = class_prob
        if self.box_thr is not None:
            assert box_thr in [.5, .6, .7, .8, .9]

        # Thresholding Training Examples
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None:
            assert isinstance(self.valid_ratio, float)
            self.video_infos = [
                x for x in self.video_infos
                if x['valid'][self.box_thr] / x['total_frames'] >= valid_ratio
            ]
            for item in self.video_infos:
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds
        for item in self.video_infos:
            item.pop('valid', None)
            item.pop('box_score', None)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    @staticmethod
    def downsample_array(arr, length):
        # If the length of the array is less than l, pad it with zeros
        if len(arr) < length:
            padded_arr = np.zeros((length, arr.shape[1], arr.shape[2]))
            padded_arr[:len(arr)] = arr
            return padded_arr

        # Calculate the step size for downsampling
        step_size = max(arr.shape[0] // (length - 1), 1)

        # Perform downsampling
        downsampled_arr = arr[::step_size][:length]

        return downsampled_arr

    def load_pkl_annotations(self):
        with open(self.ann_file, 'rb') as f:
            data = pickle.load(f)
        data_filtered = []
        counter = 0
        data_lens = [[], []]
        for item in data:
            take = True
            # Sometimes we may need to load anno from the file
            if self.split is not None:
                if item['split'] == self.split:
                    take = True
                else:
                    take = False
            if self.part is not None and take:
                if item['part'] == self.part:
                    take = True
                else:
                    take = False
            if take:
                if item['keypoint'].shape[0] > 0:
                    kp = item['keypoint'][0]
                    kp = np.where(kp > 3, 0, kp)
                    kp = np.where(kp < -3, 0, kp)
                    # kp -= np.sum(np.sum(kp, axis=1), axis=0) / np.count_nonzero(np.sum(kp, axis=-1))
                    item['keypoint'][0] = kp
                    if kp.shape[1] != 11:
                        idx = [13, 11, 4, 9, 23, 18, 25, 12, 14, 5, 19]
                        # idx = [14, 4, 11, 21, 13, 25, 7, 5, 12, 3, 18, 23, 9]
                        item['keypoint'] = item['keypoint'][:, :, :len(idx), :]
                        if self.length is None:
                            for i in range(len(idx)):
                                item['keypoint'][0, :, i] = kp[:, idx[i]]
                        else:
                            kp_ = np.zeros((self.length, len(idx), 3))
                            if len(kp) > self.length:
                                kp_red = kp_.copy()
                                for i in range(len(idx)):
                                    kp_red[:, i] = kp[:, idx[i]]
                                kp_ = kp_red[:self.length, :, :]
                                item['keypoint'][0] = kp_
                            else:
                                kp_red = np.zeros((len(kp), len(idx), 3))
                                for i in range(len(idx)):
                                    kp_red[:, i] = kp[:, idx[i]]
                                kp_[:len(kp), :, :] = kp_red
                                kp_ = kp_[np.newaxis, :]
                                item['keypoint'] = kp_
                        # if self.part == 'test':
                        #     bd.save_scatter_plot_with_names(counter, item['keypoint']
                        #     [0, 0, :, 0], item['keypoint'][0, 0, :, 1])
                        #     counter += 1
                        # item['keypoint'][0] = keypoints_reduced
                    """
                    if item['keypoint'].sum() > 0:
                        # if item['keypoint'][0].shape[0] != self.length:
                        #     raise ValueError
                        ske = item['keypoint'][0]
                        ske_new = np.zeros_like(ske)
                        while ske_new.any() != ske.any():
                            ske_new = joint_filling_np(ske)
                        ske = ske_new
                        ske = joint_smoothing_np(ske)
                        # ske = np.sum(np.sum(ske, axis=1), axis=0) / np.count_nonzero(np.sum(ske, axis=-1))
                        item['keypoint'][0] = ske
                    if counter % 1000 == 0:
                        print(counter, '/', data_len, '(', int(counter / data_len * 100), ')')
                    """
                    kp = item['keypoint'][0]
                    # Flatten the array along the first two axes
                    flattened = np.ravel(np.sum(np.sum(kp, axis=-1), axis=-1))

                    # Find the index of the first zero value
                    index = np.argwhere(np.abs(flattened) > 0.01)[:, 0]

                    mask = np.where(item['keypoint'][0] == 0., 0, 1)
                    if self.part == 'train':
                        data_lens[0].append(len(index))
                    elif self.part == 'test':
                        data_lens[1].append(len(index))
                    # non_zero_values = item['keypoint'][0][index]
                    non_zero_mean = np.sum(np.sum(item['keypoint'][0], 0), 0) / np.sum(mask)
                    mean_sub = non_zero_mean * mask
                    item['keypoint'][0] -= mean_sub
                    kp = item['keypoint'][0][index]
                    item['keypoint'] = kp[np.newaxis, :]

                    if np.sum(item['keypoint']) > 0.:
                        if item['keypoint'][0].shape[1] != 11:
                            print(item['keypoint'][0].shape)
                        data_filtered.append(item)
                        counter += 1
        return data_filtered


def smooth_sequence(sequence, max_window_size=5):
    """
    Smooth a skeleton sequence using a dynamically increasing moving average window.

    Parameters:
    - sequence (numpy array): Input skeleton sequence with shape (B, T, V, C).
    - max_window_size (int): Maximum size of the moving average window.

    Returns:
    - smoothed_sequence (numpy array): Smoothed skeleton sequence.
    """

    B, T, V = sequence.shape
    smoothed_sequence = np.zeros_like(sequence)

    for b in range(B):
        for v in range(V):
            for t in range(T):
                # Dynamically adjust window size based on the position in the sequence
                window_size = min(t + 1, T - t, max_window_size)
                smoothed_sequence[b, t, v] = np.convolve(
                    sequence[b, :, v], np.ones(window_size) / window_size, mode='same'
                )[t]

    return smoothed_sequence


if __name__ == '__main__':
    ann_file = '/home/dav86141/data/daa_data/driveandact_12.pkl'
    dataset = DAADataset(ann_file, pipeline=None, split='train', part=1)
    print(dataset.__len__)
    tmp = dataset[0]
