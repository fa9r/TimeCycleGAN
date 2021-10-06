"""Script to train models"""

import os
import datetime

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from timecyclegan.models import get_model, get_task, calculate_block_size
from timecyclegan.dataset.video_dataset import VideoDataset
from timecyclegan.dataset.concat_dataset import ConcatDataset
from timecyclegan.util.os_utils import make_dir
from timecyclegan.util.argparser import parse_args
from timecyclegan.util.logging import log_command, log_traininig_to_csv


def split_kwargs(spatial_scale, temporal_scale, **kwargs):
    """
    define keyword arguments for dataset, define_dataset(), dataloader, model, and train()
    :param kwargs: argparser arguments from which the kwargs will be built
    :param spatial_scale: current spatial scale; will be used to set width/height accordingly
    :param temporal_scale: current temporal scale; will be used to set block_size accordingly
    :return: (dataset_kwargs, dataset_def_kwargs, dataloader_kwargs, model_kwargs, train_kwargs)
    """
    # set load size for cropping; if no (or too small) load sizes are provided, set to image size (so don't random crop)
    load_width = max(kwargs["load_width"], kwargs["image_width"])
    load_height = max(kwargs["load_height"], kwargs["image_height"])
    # spatial/temporal scaling
    width = int(kwargs["image_width"] * spatial_scale)
    height = int(kwargs["image_height"] * spatial_scale)
    load_width = int(load_width * spatial_scale)
    load_height = int(load_height * spatial_scale)
    # calculate block_size
    block_size = calculate_block_size(kwargs["model_type"], kwargs["block_size"], is_train=True)
    block_size = int(block_size * temporal_scale)
    batch_size = int(kwargs["batch_size"] * 2 ** np.ceil(np.log2(1 / temporal_scale)))
    print("Sequence Length:", block_size)
    print("Batch Size:", batch_size)
    dataset_kwargs = {
        'block_size': block_size,
        'width': width,
        'height': height,
        'load_width': load_width,
        'load_height': load_height,
        'source_dir': kwargs["source_dir"],
        'target_dir': kwargs["target_dir"],
        'unpaired': get_task(kwargs["model_type"]) == "unpaired"
    }
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': kwargs["threads"],
        'pin_memory': (kwargs["gpu"] >= 0),
        'drop_last': True,
    }
    model_kwargs = {
        'model_type': kwargs["model_type"],
        'block_size': block_size,
        'gan_mode': kwargs["gan_mode"],
        'generator_architecture': kwargs["generator_architecture"],
        'generator_filters': kwargs["generator_filters"],
        'discriminator_architecture': kwargs["discriminator_architecture"],
        'discriminator_filters': kwargs["discriminator_filters"],
        'is_train': True,
        'num_epochs': kwargs["num_epochs"],
        'gpu_ids': None if kwargs["gpu"] < 0 else [kwargs["gpu"]],
        'learning_rate': kwargs["learning_rate"],
        'width': width,
        'height': height,
        'batch_size': batch_size,
        'paired_kwargs': {
            'l1_loss_weight': kwargs["l1_loss_weight"],
            'feature_matching_loss_weight': kwargs["feature_matching_loss_weight"],
            'perceptual_loss_weight': kwargs["perceptual_loss_weight"],
        },
        'unpaired_kwargs': {
            'cycle_loss_weight': kwargs["cycle_loss_weight"],
        },
        'sequential_kwargs': {
            'n_frames_G': kwargs["n_frames_generator"],
        },
        'sequence_discriminator_kwargs': {
            'n_frames_D': kwargs["n_frames_discriminator"] if kwargs["n_frames_discriminator"] > 0 else block_size,
            'temporal_scales': kwargs["discriminator_temporal_scales"],
        },
        'recyclegan_kwargs': {
            'recycle_loss_weight': kwargs["recycle_loss_weight"],
            'recycle_predictor_architecture': kwargs["recycle_predictor_architecture"],
            'recycle_predictor_filters': kwargs["recycle_predictor_filters"],
        },
        'vid2vid_kwargs': {
            'flow_loss_weight': kwargs["flow_loss_weight"],
            'mask_loss_weight': kwargs["mask_loss_weight"],
        },
        'warp_kwargs': {
            'warp_loss_weight': kwargs["warp_loss_weight"],
        },
        'timecycle_kwargs': {
            'timecycle_type': kwargs["timecycle_type"],
            'timecycle_loss': kwargs["timecycle_loss"],
            'timecycle_loss_weight': kwargs["timecycle_loss_weight"],
            'timecycle_motion_model_architecture': kwargs["timecycle_motion_model_architecture"],
            'timecycle_motion_model_filters': kwargs["timecycle_motion_model_filters"],
            'timecycle_separate_motion_models': kwargs["timecycle_separate_motion_models"],
            'timecycle_warp_loss_weight': kwargs["timecycle_warp_loss_weight"],
        },
    }
    train_kwargs = {
        'log_losses_every': kwargs["log_every"],
        'log_images_every': kwargs["log_images_every"] if kwargs["log_images_every"] > 0 else kwargs["log_every"],
        'save_every': kwargs["save_every"],
    }
    return dataset_kwargs, dataloader_kwargs, model_kwargs, train_kwargs


def get_train_hparams(spatial_scaling, temporal_scaling, **kwargs):
    """
    Build a dict of train hyperparameters (from args),
    which we would like to log to Tensorboard.
    """
    return {
        "Name": kwargs["model_name"],
        "Type": kwargs["model_type"],
        "Source": kwargs["source_dir"],
        "Target": kwargs["target_dir"],
        "Batch Size": kwargs["batch_size"],
        "Epochs": kwargs["num_epochs"],
        "TemporalScalingStart": temporal_scaling[0],
        "TemporalScalingEnd": temporal_scaling[-1],
        "TemporalScalingSteps": len(temporal_scaling),
        "SpatialScalingStart": spatial_scaling[0],
        "SpatialScalingEnd": spatial_scaling[-1],
        "SpatialScalingSteps": len(spatial_scaling),
    }


def define_train_dataset(source_dir, target_dir, unpaired, block_size, **dataset_kwargs):
    """
    Define the training dataset
    :param source_dir: source image root directory
    :param target_dir: target image root directory
    :param unpaired: whether we do paired (False) or unpaired (True) training
    :param block_size: load data in frame-blocks of size block_size
    :param dataset_kwargs: further keyword args for VideoDataset
    :return: PyTorch dataset
    """
    root_dirs = [source_dir, target_dir]  # define root directories
    root_names = ['source', 'target']  # define names of root directories
    if unpaired:
        source_dataset = VideoDataset([source_dir], ['source'], block_size, **dataset_kwargs)
        target_dataset = VideoDataset([target_dir], ['target'], block_size, **dataset_kwargs)
        dataset = ConcatDataset(source_dataset, target_dataset)
    else:
        dataset = VideoDataset(root_dirs, root_names, block_size, **dataset_kwargs)
    return dataset


def train_model(
        model,
        dataloader,
        checkpoint_dir,
        global_step=0,
        init_epoch=0,
        num_epochs=200,
        log_losses_every=100,
        log_images_every=100,
        save_every=1,
):
    """
    Train a given model on images loaded by a dataloader
    :param model: model to be trained
    :param dataloader: train dataloader (torch.utils.data.Dataloader)
    :param checkpoint_dir: directory to which model checkpoints will be saved
    :param global_step: Global training iteration (includes scaling, ...); used for printing iteration count
    :param init_epoch: If set, resume training from init_epoch
    :param num_epochs: number of training epochs
    :param log_losses_every: log losses to Tensorboard every log_losses_every iterations
    :param log_images_every: log images to Tensorboard every log_images_every iterations
    :param save_every: save model checkpoints every save_every epochs
    """
    for epoch in range(init_epoch + 1, num_epochs + 1):
        print("----------", datetime.datetime.now(), "Epoch", epoch, "----------")
        for i, batch in enumerate(dataloader):
            model.set_input(batch)  # unpack data from dataset
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            model.update_step_and_losses()  # calculate running mean for all losses
            global_step += 1  # total train iteration, starting from 1
            if global_step % log_losses_every == 0:
                print(datetime.datetime.now(), "Iteration", global_step)
                model.log_losses()  # log avg losses
            if global_step % log_images_every == 0:
                model.log_images()  # log images
        model.save(checkpoint_dir)  # save most recent epoch
        if epoch % save_every == 0:  # save additional separate checkpoints every save_every epochs
            model.save(checkpoint_dir, epoch)


def train(
        init_epoch=None,
        init_checkpoint_dir=None,
        temporal_scaling=None,
        spatial_scaling=None,
        **kwargs
):
    """Main function - solver"""
    model_name, num_epochs = kwargs["model_name"], kwargs["num_epochs"]
    checkpoint_dir = make_dir(os.path.join('checkpoints', model_name))
    writer = SummaryWriter(os.path.join("runs", model_name))
    global_step, global_epoch = 0, 0
    scales = [(spatial_scale, temporal_scaling[0]) for spatial_scale in spatial_scaling]
    scales += [(spatial_scaling[-1], temporal_scale) for temporal_scale in temporal_scaling[1:]]
    for spatial_scale, temporal_scale in scales:
        if len(spatial_scaling) > 1 or len(temporal_scaling) > 1:
            print("---------- Progressive Spatio-temporal Growing ----------\n"
                  "Spatial Scale: ", spatial_scale, "\nTemporal Scale:", temporal_scale)
        dataset_kwargs, dataloader_kwargs, model_kwargs, train_kwargs = split_kwargs(
            spatial_scale,
            temporal_scale,
            **kwargs
        )
        dataset = define_train_dataset(**dataset_kwargs)
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        steps_per_epoch = int(len(dataset) / dataloader_kwargs["batch_size"])
        num_epochs = num_epochs
        if global_step == 0:
            if init_epoch > 0:
                global_step = int(init_epoch * steps_per_epoch)
                global_epoch = init_epoch
                num_epochs -= init_epoch
            model = get_model(
                global_step=global_step,
                init_epoch=init_epoch,
                **model_kwargs
            )
            model.connect_tensorboard_writer(writer)
            train_hparams = get_train_hparams(
                spatial_scaling=spatial_scaling,
                temporal_scaling=temporal_scaling,
                **kwargs
            )
            hparams = model.log_hparams(train_hparams)  # log hyperparameters
            log_traininig_to_csv(hparams=hparams)
            model.print_params()  # print model param counts
            if init_checkpoint_dir or init_epoch > 0:
                init_checkpoint_dir = init_checkpoint_dir if init_checkpoint_dir else checkpoint_dir
                model.load(init_checkpoint_dir, epoch=init_epoch)  # load saved model weights
        else:
            model = get_model(global_step=global_step, init_epoch=0, **model_kwargs)
            model.connect_tensorboard_writer(writer)
            model.load(checkpoint_dir)
        train_model(
            model=model,
            dataloader=dataloader,
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
            init_epoch=global_epoch,
            num_epochs=global_epoch+num_epochs,
            **train_kwargs
        )
        global_step += int(num_epochs * steps_per_epoch)
        global_epoch += num_epochs


if __name__ == '__main__':
    kwargs = parse_args(mode="train")
    log_command()
    train(**kwargs)
