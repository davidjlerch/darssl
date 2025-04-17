import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler
from tqdm import tqdm
import time
import wandb
import yaml
import random
from typing import Tuple, Optional
from sklearn.metrics import balanced_accuracy_score
from datasets.nucla_dataset import NUCLADataset
from datasets.ntu_dataset import NTUDataset
from datasets.daa_dataset import DAADataset
from datasets.pku_mmd_dataset import PKUMMDDataset
from models.transformer import TransformerModel
from models.sttformer import Model as STTFormer
from models.st_gcn import Model as STGCN
from common import metric_tracking as mt
from common import utils, lr_scheduler
from common.parser import load_config, parse_args
from common.default_config import assert_and_infer_cfg
from models import custom_losses as cl
from models.classifier import LinearClassifier, StackedModel
from datasets.preprocess import augmentations
from torch.multiprocessing import set_sharing_strategy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata


set_sharing_strategy('file_system')

CKPT_PATH = 'checkpoints_lc'
CKPT_BEST_FNAME = 'checkpoint_best_lc'
TRAINING_ARGS = 'training_args_lc.yaml'


class BalancedDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, *, prefetch_factor=2,
                 persistent_workers=False):
        if sampler is None:
            sampler = BalancedSampler(dataset)

        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers)


class BalancedSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples))
        self.class_counts = self._get_class_counts()
        # Shuffle indices for each class
        class_indices = {label: [] for label in self.class_counts}
        for idx in self.indices:
            label = torch.argmax(self.dataset[idx]['label']).item()  # Assuming labels are in the second position of each sample tuple
            class_indices[label].append(idx)

        # Calculate the minimum class count
        self.min_count = min(self.class_counts.values())

    def _get_class_counts(self):
        class_counts = {}
        for idx in range(self.num_samples):
            label = torch.argmax(self.dataset[idx]['label']).item()  # Assuming labels are in the second position of each sample tuple
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        return class_counts

    def __iter__(self):
        # Sample indices equally from each class
        sampled_indices = []
        for _ in range(self.__len__() // self.min_count // self.dataset.num_classes):
            for label, count in self.class_counts.items():
                sampled_indices.extend(torch.randperm(count)[:self.min_count])

        # Shuffle the sampled indices
        sampled_indices = torch.randperm(len(sampled_indices)).tolist()

        return iter([sampled_indices[i] for i in range(len(sampled_indices))])

    def __len__(self):
        return len(self.indices)


def init_models(cfg):
    classifier = LinearClassifier(cfg.MODEL.HIDDEN_DIM, cfg.DATA.CLASSES, dropout=cfg.MODEL.DROPOUT)
    if cfg.MODEL.TRANSFORMER_ENABLE:
        if cfg.MODEL.ARCHITECTURE == 'TRANSFORMER':
            model = TransformerModel(
                orig_dim=cfg.MODEL.ORIG_DIM, hidden_dim=cfg.MODEL.HIDDEN_DIM,
                depth=cfg.MODEL.DEPTH, num_heads=cfg.MODEL.NUM_HEADS,
                mlp_ratio=cfg.MODEL.MLP_RATIO, autoregressive=cfg.MODEL.AUTOREGRESSIVE, model_type=cfg.MODEL.TYPE,
                embed_mask=cfg.AUGMENTATION.FRAMES_MASK_PROPABILITY, num_classes=cfg.DATA.CLASSES)
        elif cfg.MODEL.ARCHITECTURE == 'STTFORMER':
            model = STTFormer(6, cfg.MODEL.HIDDEN_DIM, cfg.MODEL.ORIG_DIM // 3,
                              cfg.DATA.LENGTH, 4, 1, 3,
                              [1, 1], use_pes=True, config=[[64,  64,  16], [64,  64,  16],
                                                            [64, 128, 32], [128, 128, 32],
                                                            [128, 256, 64], [256, 256, 64],
                                                            [256, 256, 64], [256, 256, 64]],
                              att_drop=0, dropout=cfg.MODEL.DROPOUT, dropout2d=0)
        elif cfg.MODEL.ARCHITECTURE == 'STGCN':
            model = STGCN(in_channels=3, out_channels=cfg.MODEL.HIDDEN_DIM, num_class=cfg.DATA.CLASSES,
                          graph_args={'layout': cfg.DATA.SET + str(cfg.MODEL.ORIG_DIM // 3)},
                          edge_importance_weighting=False, dropout=cfg.MODEL.DROPOUT)
        else:
            raise NotImplementedError
    return model, model, model


def create_ckpt_path(cfg) -> Tuple[str, str, str]:
    time_cur = time.strftime('%Y%m%d-%H:%M:%S')
    experiment_name = f'bs{cfg.TRAIN.BATCH_SIZE}_lr{cfg.OPTIMIZER.LR}_' \
                      f'seed{cfg.SEED}_{time_cur}'
    if cfg.NOTE is not None:
        experiment_name = f'{cfg.NOTE}_{experiment_name}'
    ckpt_path = os.path.join(CKPT_PATH, experiment_name)
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_save_path = os.path.join(ckpt_path, CKPT_BEST_FNAME)
    return experiment_name, ckpt_path, ckpt_save_path


def store_checkpoint(epoch: int, model, optimizer, logger, fpath: Optional[str] = None,
                     lr_scheduler=None) -> None:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler':
            lr_scheduler.state_dict()
            if lr_scheduler is not None
            else None,
        'epoch': epoch,
    }
    logger.info(f'Storing ckpt at epoch {epoch} to {fpath}')
    torch.save(checkpoint, fpath)


def get_statistics(dataloader, cfg):
    labels = []
    for i, data in enumerate(tqdm(dataloader)):
        label = data['label']
        label = torch.argmax(label, dim=1)
        labels.append(label.flatten())
    hist, bins = np.histogram(np.concatenate(labels).flatten(), bins=cfg.DATA.CLASSES)
    bins = np.arange(0, cfg.DATA.CLASSES)
    return hist, bins


def reshape_sequence(sequence):
    # Reshape the sequence to B, T, V, C
    return sequence.reshape(sequence.shape[0], sequence.shape[1], -1, 3)[:, :, :, :2]


def update(frame, batch, reshaped_sequence, colors, neighbor_link, ax):
    ax.clear()

    ax.set_xlim([-0.5, 1])
    ax.set_ylim([-0.75, 0.75])

    current_sequence = reshaped_sequence[batch, frame, :, :]

    # Get joint indices
    joint_indices = np.arange(current_sequence.shape[0])

    # Scatter plot with different colors for each joint
    scatter = ax.scatter(current_sequence[:, 0], current_sequence[:, 1], color=colors)

    # Draw connections (bones)
    for link in neighbor_link:
        x = [current_sequence[link[0], 0], current_sequence[link[1], 0]]
        y = [current_sequence[link[0], 1], current_sequence[link[1], 1]]
        ax.plot(x, y, color='black', linestyle='-', linewidth=2)

    # Annotate each point with its index
    for i, txt in enumerate(joint_indices):
        ax.annotate(txt, (current_sequence[i, 0], current_sequence[i, 1]), textcoords="offset points",
                     xytext=(0, 5), ha='center')

    ax.set_title(f'2D Plot for Batch {batch}, Timestamp {frame}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid(True)

    return scatter,


def save_2d_plot(sequence, batch, output_path):
    # sequence: B, T, V, C
    # batch: int
    # output_path: str

    reshaped_sequence = reshape_sequence(sequence)

    # Get every 10th timestamp
    timestamps = np.arange(0, reshaped_sequence.shape[1], 80)

    # Create a color map for different joints
    colors = cm.rainbow(np.linspace(0, 1, reshaped_sequence.shape[2]))

    # Create a 2D plot
    plt.figure(figsize=(8, 8))

    for timestamp in timestamps:
        # Select the specific batch and timestamp
        current_sequence = reshaped_sequence[batch, timestamp, :, :]

        # Get joint indices
        joint_indices = np.arange(current_sequence.shape[0])

        # Scatter plot with different colors for each joint
        for i in range(current_sequence.shape[0]):
            plt.scatter(current_sequence[i, 0], current_sequence[i, 1], color=colors[i], label=f'Joint {i}')

        # Draw connections (bones)
        neighbor_1base = [(1, 5), (1, 4), (1, 8), (1, 9), (2, 3), (3, 4),
                          (5, 6), (6, 7), (9, 10), (9, 11)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]

        for link in neighbor_link:
            x = [current_sequence[link[0], 0], current_sequence[link[1], 0]]
            y = [current_sequence[link[0], 1], current_sequence[link[1], 1]]
            plt.plot(x, y, color='black', linestyle='-', linewidth=2)

        # Annotate each point with its index
        for i, txt in enumerate(joint_indices):
            plt.annotate(txt, (current_sequence[i, 0], current_sequence[i, 1]), textcoords="offset points",
                         xytext=(0, 5), ha='center')

    plt.title(f'2D Plot for Batch {batch}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.grid(True)

    # Save the plot
    plt.savefig(output_path.format(batch=batch))
    plt.close()


def create_animation(sequence, batch=0, win_size=1, is_training=True):
    print('Creating Animation')
    reshaped_sequence = reshape_sequence(sequence)

    # Create a color map for different joints
    colors = cm.rainbow(np.linspace(0, 1, reshaped_sequence.shape[2]))

    # Define connections (bones)
    neighbor_1base = [(1, 5), (1, 4), (1, 8), (1, 9), (2, 3), (3, 4),
                      (5, 6), (6, 7), (9, 10), (9, 11)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim([-0.5, 1])
    ax.set_ylim([-0.75, 0.75])

    # Create animation
    animation = FuncAnimation(fig, update, frames=reshaped_sequence.shape[1],
                              fargs=(batch, reshaped_sequence, colors, neighbor_link, ax),
                              interval=100, blit=True)
    if is_training:
        animation.save('2d_plot_animation_' + str(batch) + '_' + str(win_size) + 't.gif', writer='imagemagick')
    else:
        animation.save('2d_plot_animation_' + str(batch) + '_' + str(win_size) + 'v.gif', writer='imagemagick')


def smooth_sequence(sequence, max_window_size=5):
    """
    Smooth a skeleton sequence using a dynamically increasing moving average window.

    Parameters:
    - sequence (numpy array): Input skeleton sequence with shape (B, T, V, C).
    - max_window_size (int): Maximum size of the moving average window.

    Returns:
    - smoothed_sequence (numpy array): Smoothed skeleton sequence.
    """
    print('Smoothing Sequence')
    B, T, V = sequence.shape
    smoothed_sequence = np.zeros_like(sequence)
    sequence = interpolate_3d_array(sequence)
    for b in range(B):
        for v in range(V):
            for t in range(T):
                # Dynamically adjust window size based on the position in the sequence
                window_size = min(t + 1, T - t, max_window_size)
                smoothed_sequence[b, t, v] = np.convolve(
                    sequence[b, :, v], np.ones(window_size) / window_size, mode='same'
                )[t]

    return smoothed_sequence


def interpolate_3d_array(array):
    """
    Interpolate a 3D array with missing values (0).

    Parameters:
    - array (numpy array): Input 3D array with shape (X, Y, Z).

    Returns:
    - interpolated_array (numpy array): Interpolated 3D array.
    """

    shape = array.shape
    array = np.asarray(array)
    non_zero_indices = np.array(np.where(array != 0)).T
    values = array[array != 0]

    # Generate grid for the interpolation
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

    # Interpolate missing values
    interpolated_values = griddata(non_zero_indices, values, (grid_y, grid_x, grid_z), method='linear', fill_value=0)

    return interpolated_values


def interpolate_3d_array_torch(array):
    """
    Interpolate a 3D array with missing values (0) using PyTorch's grid_sample.

    Parameters:
    - array (numpy array): Input 3D array with shape (X, Y, Z).

    Returns:
    - interpolated_array (numpy array): Interpolated 3D array.
    """

    # Convert to PyTorch tensor
    array_tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Generate grid for the interpolation
    x = torch.linspace(0, array.shape[1] - 1, array.shape[1])
    y = torch.linspace(0, array.shape[0] - 1, array.shape[0])
    z = torch.linspace(0, array.shape[2] - 1, array.shape[2])

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).float()

    # Normalize grid to [-1, 1]
    grid_normalized = (grid / torch.Tensor([array.shape[1] - 1, array.shape[0] - 1, array.shape[2] - 1]) * 2) - 1

    # Use grid_sample for interpolation
    interpolated_tensor = F.grid_sample(array_tensor, grid_normalized.unsqueeze(0))

    # Convert back to numpy array
    interpolated_array = interpolated_tensor.squeeze().numpy()

    return interpolated_array


def run_one_epoch(
        cfg, epoch, transformer, dataloader, loss_fn, optim,
        metric_tracker, device, is_training, scheduler, classifier, label_weights=None) -> None:
    y_true = []
    y_pred = []
    cls_loss = []
    s = np.random.randint(0, len(dataloader))
    for i, data in enumerate(tqdm(dataloader)):
        # create_animation(smooth_sequence(data['input'], max_window_size=5), 25, 5)
        # create_animation(smooth_sequence(data['input'], max_window_size=15), 25, 15)
        # raise ValueError
        if is_training:
            enc_in = data['input'].to(device)
        else:
            enc_in = data['keypoint'].to(device)
        label = data['label'].to(device)
        total_frames = data['total_frames'].to(device)
        forward_fn = transformer.forward
        if isinstance(transformer, TransformerModel):
            mem, tfm_out = forward_fn(enc_in, total_frames, cfg.MODEL.MEMORY_TYPE)
        elif isinstance(transformer, STTFormer):
            mem, tfm_out = forward_fn(enc_in)
        elif isinstance(transformer, STGCN):
            C = 3
            M = 1
            N, T, VC = enc_in.shape
            enc_in = enc_in.view(N, T, VC // C, C, M).permute(0, 3, 1, 2, 4)
            mem, tfm_out = forward_fn(enc_in)
        else:
            raise NotImplementedError

        cls_in = mem[:, 0, :]
        cls = cls_in
        prd = torch.argmax(cls, dim=1).to(device)
        gtr = torch.argmax(label, dim=1).to(device)
        classifier_loss = loss_fn(cls, label)
        cls_, label_ = prd.cpu(), gtr.cpu()
        y_true.append(label_)
        y_pred.append(cls_)

        if is_training:
            optim.zero_grad()
            classifier_loss.backward()
            optim.step()
            scheduler.step(epoch)
        cls_loss.append(classifier_loss.clone().detach().item())

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    if cfg.DATA.SET == 'DAA':
        classifier_acc = balanced_accuracy_score(y_true, y_pred)
        print((y_pred == y_true).sum() / len(y_pred))
    else:
        classifier_acc = (y_pred == y_true).sum() / len(y_pred)

    metrics = {
        'classifier_loss': sum(cls_loss) / len(cls_loss),
        'classifier_acc': classifier_acc
    }

    metric_tracker.update(metrics, batch_size=enc_in.size(0), is_training=is_training)


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    cfg = load_config(args, args.cfg_file)
    cfg = assert_and_infer_cfg(cfg)
    set_seed(cfg.SEED)

    experiment_name, ckpt_path, save_path = create_ckpt_path(cfg)
    with open(f'{ckpt_path}/{TRAINING_ARGS}', 'w') as f:
        yaml.dump(utils.cfg2dict(cfg), f, default_flow_style=False)

    logger = utils.get_logger(ckpt_path, level=cfg.LOG_LEVEL)

    if cfg.DATA.SET == 'DAA' or cfg.DATA.SET == 'NTU_DAA':
        normalizer = augmentations.Normalize3D(scale=None, zaxis=[0, 7], xaxis=[4, 3],
                                               dataset="DAA", align_spine=False, align_center=False)
    elif cfg.DATA.SET == 'NTU' or cfg.DATA.SET == 'PKU':
        normalizer = augmentations.Normalize3D(scale=cfg.PREPROCESSING.SCALE)
    elif cfg.DATA.SET == 'NUCLA':
        normalizer = augmentations.Normalize3D(scale=cfg.PREPROCESSING.SCALE, zaxis=[8, 4], xaxis=[0, 1])
    joint_masker = augmentations.RandomMasking(augmentations={'joints': cfg.AUGMENTATION.JOINTS_MASK_PROPABILITY})
    frame_masker = augmentations.RandomMasking(augmentations={'frames': cfg.AUGMENTATION.FRAMES_MASK_PROPABILITY})
    noiser = augmentations.RandomAdditiveNoise(dist=cfg.AUGMENTATION.NOISE_TYPE,
                                               prob=cfg.AUGMENTATION.NOISE_PROPABILITY,
                                               std=cfg.AUGMENTATION.NOISE_STD)
    rotate = augmentations.RandomRot(theta=0.3)
    augmentation = torchvision.transforms.Compose([joint_masker, frame_masker, rotate])
    noise_t = augmentations.RandomAdditiveNoise(dist=cfg.AUGMENTATION.NOISE_TYPE,
                                                prob=0.0,
                                                std=0.00)

    if cfg.DATA.SET == 'NTU':
        trainset = NTUDataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer, augmentation=augmentation,
            split=cfg.DATA.SPLIT + '_train', multi_class=True, num_classes=cfg.DATA.CLASSES)
    elif cfg.DATA.SET == 'DAA':
        trainset = DAADataset(cfg.DATA.DATA_PATH, pipeline=normalizer, augmentation=augmentation, frame_masker=frame_masker,
                              joint_masker=joint_masker, noiser=noiser, length=cfg.DATA.LENGTH,
                              split=cfg.DATA.PART, part='train', num_classes=cfg.DATA.CLASSES)
    elif cfg.DATA.SET == 'PKU':
        trainset = PKUMMDDataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer, augmentation=augmentation,
            split=cfg.DATA.SPLIT + '_train', multi_class=True, num_classes=cfg.DATA.CLASSES)
    elif cfg.DATA.SET == 'NUCLA':
        trainset = NUCLADataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer, augmentation=augmentation,
            split=cfg.DATA.SPLIT + '_train', multi_class=True, num_classes=cfg.DATA.CLASSES)

    train_loader = BalancedDataLoader(
        trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=10,
        collate_fn=trainset.collate_fn,
    )

    if cfg.DATA.SET == 'NTU':
        valset = NTUDataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer, split=cfg.DATA.SPLIT + '_val',
            multi_class=True, num_classes=cfg.DATA.CLASSES)
    elif cfg.DATA.SET == 'DAA':
        valset = DAADataset(cfg.DATA.DATA_PATH, pipeline=normalizer, frame_masker=frame_masker,
                            joint_masker=joint_masker, noiser=noiser, augmentation=noise_t, length=cfg.DATA.LENGTH,
                            split=cfg.DATA.PART, part=cfg.DATA.SPLIT, num_classes=cfg.DATA.CLASSES)
    elif cfg.DATA.SET == 'PKU':
        valset = PKUMMDDataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer, augmentation=augmentation,
            split=cfg.DATA.SPLIT + '_val', multi_class=True, num_classes=cfg.DATA.CLASSES)
    elif cfg.DATA.SET == 'NUCLA':
        valset = NUCLADataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer, augmentation=augmentation,
            split=cfg.DATA.SPLIT + '_val', multi_class=True, num_classes=cfg.DATA.CLASSES)
    val_loader = DataLoader(
        valset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=10,
        collate_fn=valset.collate_fn,
    )

    if cfg.TRAIN.WEIGHTED:
        idx_train_lbl_list = []
        label_weights_train = np.zeros(cfg.DATA.CLASSES)
        for _, data in enumerate(tqdm(train_loader)):
            labels = data['label'].detach().clone()
            for lbl in labels:
                idx_train_lbl_list.append(torch.argmax(lbl).item())
                label_weights_train[torch.argmax(lbl).item()] += 1
        label_weights_train = 1 / label_weights_train
        label_weights_train /= sum(label_weights_train)
        label_weights_train *= len(label_weights_train)

        weighted_sampler = WeightedRandomSampler(label_weights_train[idx_train_lbl_list], len(idx_train_lbl_list))

        train_loader = DataLoader(
            trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
            num_workers=10, sampler=weighted_sampler,
            collate_fn=trainset.collate_fn,
        )

        idx_val_lbl_list = []
        label_weights_test = np.zeros(cfg.DATA.CLASSES)
        for _, data in enumerate(tqdm(val_loader)):
            labels = data['label'].detach().clone()
            for lbl in labels:
                idx_val_lbl_list.append(torch.argmax(lbl).item())
                label_weights_test[torch.argmax(lbl).item()] += 1
        label_weights_test = 1 / label_weights_test
        label_weights_test /= sum(label_weights_test)
        label_weights_test *= len(label_weights_test)
        print(label_weights_test)

    else:
        label_weights_train = None
        label_weights_test = None

    _, transformer, _ = init_models(cfg)
    device = torch.device(cfg.DEVICE)
    transformer.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    if cfg.OPTIMIZER.TYPE in ['ADAM', 'ADAMW']:
        optim_classifier = torch.optim.AdamW(transformer.parameters(), lr=cfg.OPTIMIZER.LR,
                                             weight_decay=cfg.OPTIMIZER.DECAY)
        scheduler = lr_scheduler.build_scheduler(cfg, optim_classifier,
                                                 trainset.__len__() // cfg.TRAIN.BATCH_SIZE)
    else:
        optim_classifier = torch.optim.SGD(transformer.parameters(), lr=cfg.OPTIMIZER.LR,
                                           momentum=0.9, weight_decay=cfg.OPTIMIZER.DECAY)
        scheduler = lr_scheduler.build_scheduler(cfg, optim_classifier,
                                                 trainset.__len__() // cfg.TRAIN.BATCH_SIZE)
    metric_tracker = mt.MetricTracker()

    epochs = cfg.TRAIN.EPOCHS
    best_metric_value = 0
    for ep in range(epochs):
        logger.info(f'Epoch {ep + 1} / {epochs}')
        metric_tracker.reset()

        # training
        transformer.train()
        run_one_epoch(
            cfg, ep, transformer, train_loader, loss_fn, optim_classifier, metric_tracker, device, True, scheduler,
            transformer, label_weights_train)

        # validation
        transformer.eval()
        with torch.no_grad():
            run_one_epoch(
                cfg, ep, transformer, val_loader, loss_fn, optim_classifier, metric_tracker, device, False, scheduler,
                transformer, label_weights_test)

        logger.info(metric_tracker.to_string(True))
        logger.info(metric_tracker.to_string(False))

        # store checkpoint
        metric_cur = metric_tracker.get_data(cfg.PRIMARY_METRIC, False)
        if metric_cur > best_metric_value:
            best_metric_value = metric_cur
            ckpt_fpath_transformer = f"{save_path}_classifier.pth"
            store_checkpoint(
                ep + 1, transformer, optim_classifier, logger,
                ckpt_fpath_transformer)

        if ep == 0:
            wandb.init(
                project='DARSSL',
                name=experiment_name,
                mode=None if cfg.USE_WANDB else "disabled",
            )

        wandb.log({
            **metric_tracker.get_all_data(is_training=True),
            **metric_tracker.get_all_data(is_training=False),
        })


if __name__ == '__main__':
    main()
