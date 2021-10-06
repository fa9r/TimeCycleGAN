"""Script to test/inference models"""

import os

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from timecyclegan.models import get_model, get_task, calculate_block_size
from timecyclegan.dataset.video_dataset import VideoDataset
from timecyclegan.util.os_utils import make_dir
from timecyclegan.util.argparser import parse_args
from timecyclegan.util.logging import log_command


def define_output_dir(
        model_name,
        output_root="results",
        epoch=None,
        dir_name_appendix="",
):
    """
    Define output dir to which results are written
    :param model_name: model_name of the trained model
    :param output_root: root directory of output dirs
    :param epoch: after which epoch the model was loaded
    :param dir_name_appendix: additional string to append to output_dir
    :return: path to output directory
    """
    epoch_str = ''
    if epoch:
        epoch_str = ('epoch' + str(epoch))
    return os.path.join(output_root, model_name, epoch_str, dir_name_appendix)


def split_kwargs(**kwargs):
    """
    split kwargs into dataset, dataloader, model, model loading and training
    :param kwargs: dict of argparser args to be split
    :return:
        dataset_kwargs (dict)
        dataloader_kwargs (dict)
        model_kwargs (dict)
        modelloader_kwargs (dict)
        test_kwargs (dict)
    """
    tts = kwargs["test_unpaired_target_to_source"]
    tdir = kwargs["test_target_dir"]
    sdir = kwargs["test_source_dir"]
    # define block size
    block_size = calculate_block_size(
        kwargs["model_type"],
        kwargs["block_size"],
        is_train=False
    )
    # define dir from which model is loaded
    load_dir = kwargs["model_name"]
    if not load_dir.startswith('checkpoints'):
        load_dir = os.path.join('checkpoints', load_dir)
    # define output dir to which results are written
    dir_name_appendix = ""
    if get_task(kwargs["model_type"]) == "unpaired":
        dir_name_appendix = "target_to_source" if tts else "source_to_target"
    output_dir = define_output_dir(
        output_root=kwargs["test_output_dir"],
        model_name=load_dir[12:],
        epoch=kwargs["test_init_epoch"],
        dir_name_appendix=dir_name_appendix,
    )
    dataset_kwargs = {
        'block_size': block_size,
        'root_dirs': [sdir] if not tts else [tdir],
        'root_names': ['source'] if not tts else ['target'],
        'overlap': True,
        'augment_data': False,
        'width': kwargs["image_width"],
        'height': kwargs["image_height"],
    }
    dataloader_kwargs = {
        'batch_size': kwargs["test_batch_size"],
        'shuffle': False,
        'num_workers': kwargs["threads"],
        'pin_memory': (kwargs["gpu"] >= 0),
    }
    model_kwargs = {
        'model_type': kwargs["model_type"],
        'block_size': block_size,
        'generator_architecture': kwargs["generator_architecture"],
        'generator_filters': kwargs["generator_filters"],
        'is_train': False,
        'gpu_ids': None if kwargs["gpu"] < 0 else [kwargs["gpu"]],
        'width': kwargs["image_width"],
        'height': kwargs["image_height"],
        'batch_size': kwargs["test_batch_size"],
        'unpaired_kwargs': {
            'source_to_target': not tts,
        },
        'vid2vid_kwargs': {
            'n_frames_G': kwargs["n_frames_generator"],
        },
        'timecycle_kwargs': {
            'n_frames_G': kwargs["n_frames_generator"],
        },
    }
    modelloader_kwargs = {
        'load_dir': load_dir,
        'epoch': 0 if not kwargs["test_init_epoch"] else kwargs["test_init_epoch"],
    }
    test_kwargs = {
        'input_dir': sdir if not tts else tdir,
        'output_dir': output_dir,
        'source_to_target': not tts,
    }
    return (
        dataset_kwargs,
        dataloader_kwargs,
        model_kwargs,
        modelloader_kwargs,
        test_kwargs
    )


def define_output_path(batch, input_dir, output_dir, source_to_target=True):
    """
    Define the output path to which results should be saved
    :param batch: current data batch
    :param input_dir: base directory of input (i.e. source or target root)
    :param output_dir: base directory to where results will be saved
    :param source_to_target: whether to inference source to target
        or target to source (for unpaired models)
    """
    if source_to_target:
        input_paths = batch["source_path"]
    else:
        input_paths = batch["target_path"]
    input_path = input_paths[len(input_paths) // 2][0]  # get middle image
    input_path_rel = os.path.relpath(input_path, input_dir)
    output_path = os.path.join(output_dir, input_path_rel)
    make_dir(os.path.dirname(output_path))
    return output_path


def test_model(
        model,
        dataloader,
        input_dir,
        output_dir='results',
        source_to_target=True
):
    """
    Test a given model on images loaded by a dataloader and save results
    :param model: model to be inference
    :param dataloader: test dataloader (torch.utils.data.Dataloader)
    :param output_dir: base directory to where results will be saved
    :param source_to_target: whether to inference source to target
        or target to source (for unpaired models)
    """
    last_output_dir = None
    for i, batch in enumerate(dataloader):
        # log progress to stdout
        if (i+1) % 100 == 0:
            print("Processing image [", i+1, "/", len(dataloader), "]")
        # define output path
        output_path = define_output_path(
            batch=batch,
            input_dir=input_dir,
            output_dir=output_dir,
            source_to_target=source_to_target
        )
        # reset previous geners of sequential models when a new sequence starts
        if os.path.dirname(output_path) != last_output_dir:
            model.first_test_input = True
        last_output_dir = os.path.dirname(output_path)
        # model prediction
        model.set_input(batch)  # set model input
        prediction = model.test()[0]  # run model
        prediction_unnorm = prediction / 2 + 0.5  # unnormalize results
        save_image(prediction_unnorm, output_path)


def test(**kwargs):
    """Main function"""
    (
        dataset_kwargs,
        dataloader_kwargs,
        model_kwargs,
        modelloader_kwargs,
        test_kwargs
    ) = split_kwargs(**kwargs)
    model = get_model(**model_kwargs)
    dataset = VideoDataset(**dataset_kwargs)
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    model.load(**modelloader_kwargs)
    test_model(model, dataloader, **test_kwargs)


if __name__ == '__main__':
    arg_dict = parse_args(mode="test")
    log_command()
    test(**arg_dict)
