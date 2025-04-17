import os
import sys
import getpass

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import time
from typing import Tuple, Optional
import yaml
import pickle
import random
from sklearn.metrics import multilabel_confusion_matrix, classification_report, balanced_accuracy_score

import common.Dino_utils
from datasets.nucla_dataset import NUCLADataset
from datasets.pku_mmd_dataset import PKUMMDDataset
from datasets.daa_dataset import DAADataset
from datasets.ntu_dataset import NTUDataset
from models.transformer import TransformerModel
from models.sttformer import Model as STTFormer
from models.st_gcn import Model as STGCN
from common import metric_tracking as mt
from common import utils
from common.parser import load_config, parse_args
from common.default_config import assert_and_infer_cfg
from models import custom_losses as cl
from datasets.preprocess import augmentations
from common import lr_scheduler


CKPT_PATH = 'checkpoints'
CKPT_BEST_FNAME = 'checkpoint_best'
TRAINING_ARGS = 'training_args.yaml'


def setup_ccname():
    user = getpass.getuser()
    # check if k5start is running, exit otherwise
    try:
        pid = open("/tmp/k5pid_"+user).read().strip()
        os.kill(int(pid), 0)
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nk5start not running!\n")
        sys.exit(1)
    try:
        ccname = open("/tmp/kccache_"+user).read().split("=")[1].strip()
        os.environ['KRB5CCNAME'] = ccname
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nmaybe k5start not running?\n")
        sys.exit(1)


def init_models(cfg):
    teacher, student = None, None

    assert cfg.MODEL.TYPE in ['tafar', 'smae', 'bert', 'casmae']
    if cfg.MODEL.TRANSFORMER_ENABLE:
        if cfg.MODEL.ARCHITECTURE == 'TRANSFORMER':
            student = TransformerModel(
                orig_dim=cfg.MODEL.ORIG_DIM, hidden_dim=cfg.MODEL.HIDDEN_DIM,
                depth=cfg.MODEL.DEPTH, num_heads=cfg.MODEL.NUM_HEADS,
                mlp_ratio=cfg.MODEL.MLP_RATIO, autoregressive=cfg.MODEL.AUTOREGRESSIVE, model_type=cfg.MODEL.TYPE,
                embed_mask=cfg.AUGMENTATION.FRAMES_MASK_PROPABILITY)
            teacher = TransformerModel(
                orig_dim=cfg.MODEL.ORIG_DIM, hidden_dim=cfg.MODEL.HIDDEN_DIM,
                depth=cfg.MODEL.DEPTH, num_heads=cfg.MODEL.NUM_HEADS,
                mlp_ratio=cfg.MODEL.MLP_RATIO, autoregressive=cfg.MODEL.AUTOREGRESSIVE, model_type=cfg.MODEL.TYPE,
                embed_mask=cfg.AUGMENTATION.FRAMES_MASK_PROPABILITY)
        elif cfg.MODEL.ARCHITECTURE == 'STTFORMER':
            student = STTFormer(6, cfg.MODEL.HIDDEN_DIM, cfg.MODEL.ORIG_DIM // 3,
                                90, 4, 1, 3,
                                [1, 1], use_pes=True, config=[[64,  64,  16], [64,  64,  16],
                                                              [64, 128, 32], [128, 128, 32],
                                                              [128, 256, 64], [256, 256, 64],
                                                              [256, 256, 64], [256, 256, 64]],
                                att_drop=0, dropout=cfg.MODEL.DROPOUT, dropout2d=0)
            teacher = STTFormer(6, cfg.MODEL.HIDDEN_DIM, cfg.MODEL.ORIG_DIM // 3,
                                90, 4, 1, 3,
                                [1, 1], use_pes=True, config=[[64,  64,  16], [64,  64,  16],
                                                              [64, 128, 32], [128, 128, 32],
                                                              [128, 256, 64], [256, 256, 64],
                                                              [256, 256, 64], [256, 256, 64]],
                                att_drop=0, dropout=cfg.MODEL.DROPOUT, dropout2d=0)
        elif cfg.MODEL.ARCHITECTURE == 'STGCN':
            student = STGCN(in_channels=3, out_channels=cfg.MODEL.HIDDEN_DIM, num_class=cfg.DATA.CLASSES,
                            graph_args={'layout': cfg.DATA.SET + str(cfg.MODEL.ORIG_DIM // 3)},
                            edge_importance_weighting=False)
            teacher = STGCN(in_channels=3, out_channels=cfg.MODEL.HIDDEN_DIM, num_class=cfg.DATA.CLASSES,
                            graph_args={'layout': cfg.DATA.SET + str(cfg.MODEL.ORIG_DIM // 3)},
                            edge_importance_weighting=False)
        else:
            raise NotImplementedError
        teacher.load_state_dict(student.state_dict())
    return teacher, student


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
        'optimizer':
            optimizer.state_dict()
            if optimizer is not None
            else None,
        'lr_scheduler':
            lr_scheduler.state_dict()
            if lr_scheduler is not None
            else None,
        'epoch': epoch,
    }
    logger.info(f'Storing ckpt at epoch {epoch} to {fpath}')
    torch.save(checkpoint, fpath)


def classify(embeddings, labels, classifier=None):
    if classifier:
        return classifier, classifier(embeddings)
    classifier = utils.CustomKNeighborsClassifier(embedding=embeddings, label=labels)
    return classifier, classifier(embeddings)


def run_one_epoch(
        cfg, epoch, transformer_teacher, transformer_student, dataloader, loss_fn, optim1, optim2,
        metric_tracker, device, is_training, scheduler_transformer, momentum_schedule, label_weights=None
) -> None:
    labels = []
    cls_tokens = []
    transformer_loss_sum = 0.

    for i, data in enumerate(tqdm(dataloader)):
        enc_in = data['input'].to(device)
        enc_fm = data['frame_mask'].to(device)
        enc_jm = data['joint_mask'].to(device)
        enc_noise = data['noise'].to(device)
        enc_orig = data['keypoint'].to(device)
        enc_aug = data['augment'].to(device)
        total_frames = data['total_frames'].to(device)
        enc_in = torch.zeros(enc_in.shape).to(device)
        # enc_orig = data['keypoint'].to(device)
        label = data['label'].to(device)
        # convert to a binary mask, True --> ignored during attention
        B, T, C = enc_in.size()
        forward_fn_student = transformer_student.forward
        forward_fn_teacher = transformer_teacher.forward

        # dec_in = torch.zeros(*enc_in.shape[:-1], transformer.hidden_dim).to(enc_in)
        if isinstance(transformer_student, TransformerModel):
            _, tfm_out_fm = forward_fn_student(enc_fm, total_frames, cfg.MODEL.MEMORY_TYPE)
            _, tfm_out_jm = forward_fn_student(enc_jm, total_frames, cfg.MODEL.MEMORY_TYPE)
            _, tfm_out_noise = forward_fn_student(enc_noise, total_frames, cfg.MODEL.MEMORY_TYPE)
            mem, tfm_out_s = forward_fn_student(enc_orig, total_frames, cfg.MODEL.MEMORY_TYPE)
            _, tfm_out_t = forward_fn_teacher(enc_orig, total_frames, cfg.MODEL.MEMORY_TYPE)
            _, tfm_aug_t = forward_fn_teacher(enc_aug, total_frames, cfg.MODEL.MEMORY_TYPE)
        elif isinstance(transformer_student, STTFormer):
            _, tfm_out_fm = forward_fn_student(enc_fm)
            _, tfm_out_jm = forward_fn_student(enc_jm)
            _, tfm_out_noise = forward_fn_student(enc_noise)
            mem, tfm_out_s = forward_fn_student(enc_orig)
            _, tfm_out_t = forward_fn_teacher(enc_orig)
            _, tfm_aug_t = forward_fn_teacher(enc_aug)
        elif isinstance(transformer_student, STGCN):
            C = 3
            M = 1
            N, T, VC = enc_fm.shape

            enc_fm = enc_fm.view(N, T, VC // C, C, M).permute(0, 3, 1, 2, 4)
            enc_jm = enc_jm.view(N, T, VC // C, C, M).permute(0, 3, 1, 2, 4)
            enc_noise = enc_noise.view(N, T, VC // C, C, M).permute(0, 3, 1, 2, 4)
            enc_orig = enc_orig.view(N, T, VC // C, C, M).permute(0, 3, 1, 2, 4)
            enc_aug = enc_aug.view(N, T, VC // C, C, M).permute(0, 3, 1, 2, 4)

            _, tfm_out_fm = forward_fn_student(enc_fm)
            _, tfm_out_jm = forward_fn_student(enc_jm)
            _, tfm_out_noise = forward_fn_student(enc_noise)
            mem, tfm_out_s = forward_fn_student(enc_orig)
            _, tfm_out_t = forward_fn_teacher(enc_orig)
            _, tfm_aug_t = forward_fn_teacher(enc_aug)

        else:
            raise NotImplementedError

        # print(tfm_out.shape, enc_ori<g.shape, keep_mask.shape)
        student_in = torch.cat([tfm_out_s, tfm_out_fm, tfm_out_jm, tfm_out_noise])
        teacher_in = torch.cat([tfm_out_t, tfm_aug_t])

        if label_weights is not None:
            transformer_loss = loss_fn(student_in,
                                       teacher_in, epoch, labels=label,
                                       weights=torch.from_numpy(label_weights).to(device))
        else:
            transformer_loss = loss_fn(student_in,
                                       teacher_in, epoch, labels=label)
        autoencoder_in = mem.detach().clone()
        # autoencoder_embedding, autoencoder_out = autoencoder(autoencoder_in)
        # autoencoder_loss = loss_fn(autoencoder_out, autoencoder_in)

        if transformer_loss_sum == 0.:
            transformer_loss_sum = transformer_loss.clone()

        if is_training and (i > 0 and (i % cfg.OPTIMIZER.STRIDE == 0 or i == len(dataloader))):
            optim1.zero_grad()
            # optim2.zero_grad()

            transformer_loss_sum.backward()
            # autoencoder_loss.backward()

            optim1.step()
            # optim2.step()

            scheduler_transformer.step(epoch)
            # scheduler_autoencoder.step()

            for p in transformer_teacher.parameters():
                p.requires_grad = False
            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[i]  # momentum parameter
                for param_q, param_k in zip(transformer_student.parameters(), transformer_teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            transformer_loss_sum = 0.
        else:
            if transformer_loss_sum != transformer_loss:
                transformer_loss_sum += transformer_loss

        # We take the first sequence corresponding to the cls token
        # embedding = autoencoder_embedding[:, 0]
        label = label.detach().cpu().numpy()
        embed = autoencoder_in[:, 0].detach().cpu().numpy()
        if 'part' in data:
            for s, p, l, c in zip(data['split'], data['part'], label, embed):
                p_ref = 'test'
                if is_training:
                    p_ref = 'train'
                if s.item() == cfg.DATA.PART and p == p_ref:
                    labels.append(l[np.newaxis, :])
                    cls_tokens.append(c[np.newaxis, :])
        else:
            labels.append(label)
            cls_tokens.append(embed)
        metrics = {
            'transformer_loss': transformer_loss.item()
        }

        metric_tracker.update(metrics, batch_size=enc_in.size(0), is_training=is_training)

    labels = np.concatenate(labels, axis=0).argmax(-1)
    cls_tokens = np.concatenate(cls_tokens, axis=0)
    if epoch == 0 or epoch % 10 == 9:
        if is_training:
            knc1, classes1 = classify(cls_tokens, labels)
            transformer_student.set_knc(knc1, key="cls_token")
            knnPickle = open('knnpickle_file', 'wb')
            pickle.dump(knc1, knnPickle)
            knnPickle.close()
        else:
            knc1, classes1 = classify(cls_tokens, labels, classifier=transformer_student.get_knc("cls_token"))

        if cfg.DATA.SET == 'DAA':
            prd = classes1
            gtr = labels
            y_true = gtr.flatten()
            y_pred = prd.flatten()

            acc_cls = balanced_accuracy_score(y_true, y_pred)
        else:
            acc_cls = (classes1 == labels).sum() / len(classes1)
    else:
        acc_cls = 0.

    metrics = {
        'cls_accuracy': acc_cls,
    }

    metric_tracker.update(metrics, batch_size=1, is_training=is_training)
    setup_ccname()


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
    # Prepare checkpoint path, saving training arguments
    experiment_name, ckpt_path, save_path = create_ckpt_path(cfg)
    with open(f'{ckpt_path}/{TRAINING_ARGS}', 'w') as f:
        yaml.dump(utils.cfg2dict(cfg), f, default_flow_style=False)

    logger = utils.get_logger(ckpt_path, level=cfg.LOG_LEVEL)

    # Data augmentation
    if cfg.DATA.SET == 'DAA':
        # 7, 21,  12, 13
        # normalizer = augmentations.Normalize3D(scale=cfg.PREPROCESSING.SCALE, zaxis=[8, 4], xaxis=[6, 3],
        #                                        dataset="DAA", align_spine=True, align_center=True)
        normalizer = augmentations.Normalize3D(scale=cfg.PREPROCESSING.SCALE, zaxis=[0, 7], xaxis=[4, 3],
                                               dataset="DAA", align_spine=True, align_center=True)
    elif cfg.DATA.SET == 'NTU' or cfg.DATA.SET == 'PKU':
        normalizer = augmentations.Normalize3D(scale=cfg.PREPROCESSING.SCALE)
    elif cfg.DATA.SET == 'NUCLA':
        normalizer = augmentations.Normalize3D(scale=cfg.PREPROCESSING.SCALE, zaxis=[8, 4], xaxis=[0, 1])
    noise_rot = augmentations.RandomRot()
    noise_scale = augmentations.RandomScale()
    awgn = augmentations.RandomGaussianNoise(sigma=cfg.AUGMENTATION.NOISE_STD)

    joint_masker = augmentations.RandomMasking(augmentations={'joints': cfg.AUGMENTATION.JOINTS_MASK_PROPABILITY})
    frame_masker = augmentations.RandomMasking(augmentations={'frames': cfg.AUGMENTATION.FRAMES_MASK_PROPABILITY})
    noiser = augmentations.RandomAdditiveNoise(dist=cfg.AUGMENTATION.NOISE_TYPE,
                                               prob=cfg.AUGMENTATION.NOISE_PROPABILITY,
                                               std=cfg.AUGMENTATION.NOISE_STD)
    noise_t = augmentations.RandomAdditiveNoise(dist=cfg.AUGMENTATION.NOISE_TYPE,
                                                prob=0.8,
                                                std=0.01)

    # Create dataset and dataloader
    if cfg.DATA.SET == 'NTU':
        trainset = NTUDataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer, frame_masker=frame_masker, joint_masker=joint_masker,
            noiser=noiser,
            split=cfg.DATA.SPLIT + '_train', multi_class=True, num_classes=cfg.DATA.CLASSES)
    elif cfg.DATA.SET == 'DAA':
        trainset = DAADataset(cfg.DATA.DATA_PATH, pipeline=normalizer, frame_masker=frame_masker,
                              joint_masker=joint_masker, noiser=noiser, augmentation=noise_t, length=cfg.DATA.LENGTH,
                              split=cfg.DATA.PART, part='train', num_classes=cfg.DATA.CLASSES)
    elif cfg.DATA.SET == 'PKU':
        trainset = PKUMMDDataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer, augmentation=augmentation,
            split=cfg.DATA.SPLIT + '_train', multi_class=True, num_classes=cfg.DATA.CLASSES)
    elif cfg.DATA.SET == 'NUCLA':
        trainset = NUCLADataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer, augmentation=augmentation,
            split=cfg.DATA.SPLIT + '_train', multi_class=True, num_classes=cfg.DATA.CLASSES)
    train_loader = DataLoader(
        trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
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
                            split=cfg.DATA.PART, part='test', num_classes=cfg.DATA.CLASSES)

    elif cfg.DATA.SET == 'PKU':
        valset = PKUMMDDataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer,
            split=cfg.DATA.SPLIT + '_val', multi_class=True, num_classes=cfg.DATA.CLASSES)
    elif cfg.DATA.SET == 'NUCLA':
        valset = NUCLADataset(
            cfg.DATA.DATA_PATH, pipeline=normalizer,
            split=cfg.DATA.SPLIT + '_val', multi_class=True, num_classes=cfg.DATA.CLASSES)
    val_loader = DataLoader(
        valset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=10,
        collate_fn=valset.collate_fn,
    )

    teacher, transformer = init_models(cfg)
    device = torch.device(cfg.DEVICE)
    teacher.to(device)
    transformer.to(device)

    # loss_fn = cl.SmoothL1LossWithMask(reduction='none')
    loss_fn = common.Dino_utils.DINOLoss(cfg.MODEL.HIDDEN_DIM, cfg.TRAIN.TEACHER_WARM_UP_TEMP, cfg.TRAIN.TEACHER_TEMP,
                                         cfg.TRAIN.WARMUP_EPOCHS, cfg.TRAIN.EPOCHS, ncrops=4,
                                         student_temp=cfg.TRAIN.STUDENT_TEMP, device=cfg.DEVICE)
    assert cfg.OPTIMIZER.TYPE in ['ADAM', 'SGD']
    if cfg.OPTIMIZER.TYPE == 'ADAM':
        optim_transformer = torch.optim.AdamW(transformer.parameters(), lr=cfg.OPTIMIZER.LR,
                                              weight_decay=cfg.OPTIMIZER.DECAY)
        momentum_schedule = utils.cosine_scheduler(cfg.TRAIN.CENTER_MOMENTUM, 1, cfg.TRAIN.EPOCHS,
                                                   trainset.__len__() // cfg.TRAIN.BATCH_SIZE)
        scheduler_transformer = lr_scheduler.build_scheduler(cfg, optim_transformer,
                                                             trainset.__len__() // cfg.TRAIN.BATCH_SIZE)

    else:
        optim_transformer = torch.optim.SGD(transformer.parameters(), lr=cfg.OPTIMIZER.LR,
                                            momentum=0.9, weight_decay=cfg.OPTIMIZER.DECAY)
        scheduler_transformer = lr_scheduler.build_scheduler(cfg, optim_transformer,
                                                             trainset.__len__() // cfg.TRAIN.BATCH_SIZE)
        momentum_schedule = utils.cosine_scheduler(cfg.TRAIN.CENTER_MOMENTUM, 1, cfg.TRAIN.EPOCHS,
                                                   trainset.__len__() // cfg.TRAIN.BATCH_SIZE)

    metric_tracker = mt.MetricTracker()
    label_weights = np.zeros(cfg.DATA.CLASSES)
    for _, data in enumerate(tqdm(train_loader)):
        labels = data['label'].detach().clone()
        for lbl in labels:
            label_weights[torch.argmax(lbl).item()] += 1
    label_weights = 1 / label_weights
    label_weights /= sum(label_weights)
    label_weights *= len(label_weights)
    print(label_weights)
    label_weights = None

    epochs = cfg.TRAIN.EPOCHS
    best_metric_value = 0
    for ep in range(epochs):
        logger.info(f'Epoch {ep + 1} / {epochs}')
        metric_tracker.reset()

        # training
        transformer.train()
        teacher.eval()
        run_one_epoch(
            cfg, ep, teacher, transformer, train_loader, loss_fn, optim_transformer,
            None, metric_tracker, device, True, scheduler_transformer, momentum_schedule, label_weights)

        # validation
        teacher.eval()
        transformer.eval()
        with torch.no_grad():
            run_one_epoch(
                cfg, ep, teacher, transformer, val_loader, loss_fn,
                optim_transformer, None, metric_tracker, device, False, scheduler_transformer,
                momentum_schedule, label_weights)

        logger.info(metric_tracker.to_string(True))
        logger.info(metric_tracker.to_string(False))

        # store checkpoint
        metric_cur = metric_tracker.get_data(cfg.PRIMARY_METRIC, False)
        if metric_cur > best_metric_value:
            best_metric_value = metric_cur

            # save transformer model
            ckpt_fpath_transformer = f"{save_path}_transformer.pth"
            store_checkpoint(
                ep + 1, transformer, optim_transformer, logger,
                ckpt_fpath_transformer)

            # save teacher model
            ckpt_fpath_autoencoder = f"{save_path}_autoencoder.pth"
            store_checkpoint(
                ep + 1, teacher, None, logger,
                ckpt_fpath_autoencoder)

        if ep == 0:
            wandb.init(
                project='DARSSL',
                name=experiment_name,
                mode=None if cfg.USE_WANDB else "disabled",
            )
        try:
            # wandb log info
            wandb.log({
                **metric_tracker.get_all_data(is_training=True),
                **metric_tracker.get_all_data(is_training=False),
            })
        except:
            try:
                wandb.init(
                    project='DARSSL',
                    name=experiment_name,
                    mode=None if cfg.USE_WANDB else "disabled",
                )
            except:
                pass


if __name__ == '__main__':
    main()
