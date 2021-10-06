"""
Defines all GAN models
* In networks.py the individual model networks (generators, discriminators, flownet, vgg, ...) are defined.
* BaseModel in base_model.py is an abstract base class, which other models inherit. Basic functionality is defined here.
* CycleGANModel is a reimplemented CycleGAN (see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* DCGANModel is a reimplemented DCGAN (see https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
* Pix2PixModel in pix2pix_model.py is a generalized version of Pix2Pix
    (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), which can predict entire image blocks at a time.
* RecycleGANModel is a reimplemented RecycleGAN (see https://github.com/aayushbansal/Recycle-GAN/)
* TimecycleModel in timecycle_pix2pix.py is a modified version of Pix2Pix, which uses an additional CycleGAN with cycle
    loss between subsequent frames to learn temporal consistency of generated videos.
* Vid2VidModel is a reimplemented Vid2Vid (see https://github.com/NVIDIA/vid2vid/)
"""


def calculate_block_size(model_type, block_size, is_train):
    """
    Calculate the block_size based on the model type
    :param model_type: model type ('cyclegan' | 'dcgan' | 'pix2pix' | 'recyclegan' | 'timecycle' | 'vid2vid')
    :param block_size: block_size provided by command line args (or 0 if not set)
    :param is_train: whether the function is called during training or test
    :return: model-specific block_size
    """
    if model_type in ('pix2pix', 'cyclegan', 'dcgan'):
        return 1
    if model_type == 'flownet':
        return 2
    if model_type in ('recyclegan', 'timerecyclegan'):
        return 1 if not is_train else 3
    if model_type in ('timecycle', 'tricyclegan', 'vid2vid', 'timecyclevid2vid', 'timecycledcgan', 'timerecycleganv2', 'timecyclep2'):
        return 1 if not is_train else 5 if block_size < 1 else block_size
    raise ValueError("Invalid argument: model_type '%s' not recognized." % model_type)


def get_task(model_type):
    """Get task (unconditional | paired | unpaired | other) from model type"""
    if model_type in ("dcgan", "timecycledcgan"):
        return "unconditional"
    elif model_type in ("pix2pix", "timecycle", "timecyclep2", "vid2vid", "timecyclevid2vid"):
        return "paired"
    elif model_type in ('cyclegan', 'recyclegan', 'timerecyclegan', 'timerecycleganv2', 'tricyclegan'):
        return "unpaired"
    elif model_type == "flownet":
        return "other"
    raise ValueError("Invalid argument: model_type '%s' not recognized." % model_type)


def get_model(
        model_type,
        paired_kwargs={},
        unpaired_kwargs={},
        sequential_kwargs={},
        sequence_discriminator_kwargs={},
        timecycle_kwargs={},
        warp_kwargs={},
        recyclegan_kwargs={},
        vid2vid_kwargs={},
        **base_kwargs,
):
    """
    define model based on given arguments
    :param model_type: model type ('cyclegan' | 'dcgan' | 'pix2pix' | 'recyclegan' | 'timecycle' | 'vid2vid')
    :param vid2vid_kwargs: keyword arguments for Vid2VidModel
    :param timecycle_kwargs: keyword arguments for TimecycleModel
    :param paired_kwargs: further keyword arguments for PairedModel
    :param unpaired_kwargs: further keyword arguments for UnpairedModel
    :param base_kwargs: further keyword arguments for BaseModel
    :return: model (subclass of BaseModel)
    """
    if model_type == 'cyclegan':
        from timecyclegan.models.baselines.cyclegan_model import CycleGANModel
        model = CycleGANModel(
            **unpaired_kwargs,
            **base_kwargs
        )
    elif model_type == 'dcgan':
        from timecyclegan.models.baselines.dcgan_model import DCGANModel
        model = DCGANModel(
            **base_kwargs
        )
    elif model_type == 'flownet':
        from .flownet2_model import Flownet2Model
        model = Flownet2Model(
            **paired_kwargs,
            **base_kwargs
        )
    elif model_type == 'pix2pix':
        from timecyclegan.models.baselines.pix2pix_model import Pix2PixModel
        model = Pix2PixModel(
            **paired_kwargs,
            **base_kwargs
        )
    elif model_type == 'recyclegan':
        from timecyclegan.models.baselines.recyclegan_model import RecycleGANModel
        model = RecycleGANModel(
            **unpaired_kwargs,
            **base_kwargs
        )
    elif model_type in ('timecycle', 'timecyclep2'):
        if base_kwargs["is_train"]:
            if model_type == 'timecycle':
                from .timecycle_pix2pix import TimeCyclePix2Pix
                model = TimeCyclePix2Pix(
                    **sequential_kwargs,
                    **sequence_discriminator_kwargs,
                    **timecycle_kwargs,
                    **paired_kwargs,
                    **base_kwargs
                )
            else:
                from .timecycle_pix2pix import TimeCyclePix2PixWarp
                model = TimeCyclePix2PixWarp(
                    **sequential_kwargs,
                    **sequence_discriminator_kwargs,
                    **timecycle_kwargs,
                    **warp_kwargs,
                    **paired_kwargs,
                    **base_kwargs
                )
        else:
            from .timecycle_pix2pix import SequentialPix2Pix
            model = SequentialPix2Pix(
                **sequential_kwargs,
                **paired_kwargs,
                **base_kwargs
            )
    elif model_type == 'timecycledcgan':
        if base_kwargs["is_train"]:
            from .timecycle_dcgan import TimeCycleDCGAN
            model = TimeCycleDCGAN(
                **sequential_kwargs,
                **sequence_discriminator_kwargs,
                **timecycle_kwargs,
                **base_kwargs
            )
        else:
            from .sequential import SequentialDCGAN
            model = SequentialDCGAN(
                **sequential_kwargs,
                **base_kwargs
            )
    elif model_type == 'timecyclevid2vid':
        if base_kwargs["is_train"]:
            from .timecycle_vid2vid import TimeCycleVid2Vid
            model = TimeCycleVid2Vid(
                **sequential_kwargs,
                **sequence_discriminator_kwargs,
                **timecycle_kwargs,
                **vid2vid_kwargs,
                **paired_kwargs,
                **base_kwargs)
        else:
            from timecyclegan.models.baselines.vid2vid_model import Vid2VidModel
            model = Vid2VidModel(
                **sequential_kwargs,
                **vid2vid_kwargs,
                **paired_kwargs,
                **base_kwargs,
            )
    elif model_type == 'timerecyclegan':
        if base_kwargs["is_train"]:
            from .timecycle_recyclegan import TimeCycleRecycleGAN
            model = TimeCycleRecycleGAN(
                **sequence_discriminator_kwargs,
                **timecycle_kwargs,
                **unpaired_kwargs,
                **base_kwargs
            )
        else:
            from timecyclegan.models.baselines.recyclegan_model import RecycleGANModel
            model = RecycleGANModel(
                **unpaired_kwargs,
                **base_kwargs)
    elif model_type == 'timerecycleganv2':
        if base_kwargs["is_train"]:
            from .timecycle_recyclegan import TimeCycleRecycleGANv2
            model = TimeCycleRecycleGANv2(
                **sequential_kwargs,
                **sequence_discriminator_kwargs,
                **timecycle_kwargs,
                **recyclegan_kwargs,
                **unpaired_kwargs,
                **base_kwargs
            )
        else:
            from .sequential import SequentialRecycleGAN
            model = SequentialRecycleGAN(
                **sequential_kwargs,
                **recyclegan_kwargs,
                **unpaired_kwargs,
                **base_kwargs)
    elif model_type == 'tricyclegan':
        from .timecycle_cyclegan import TimeCycleCycleGAN, SequentialCycleGAN
        if base_kwargs["is_train"]:
            model = TimeCycleCycleGAN(
                **sequential_kwargs,
                **sequence_discriminator_kwargs,
                **timecycle_kwargs,
                **unpaired_kwargs,
                **base_kwargs
            )
        else:
            model = SequentialCycleGAN(
                **sequential_kwargs,
                **unpaired_kwargs,
                **base_kwargs,
            )
    elif model_type == 'vid2vid':
        from timecyclegan.models.baselines.vid2vid_model import Vid2VidModel
        model = Vid2VidModel(
            **sequential_kwargs,
            **sequence_discriminator_kwargs,
            **warp_kwargs,
            **vid2vid_kwargs,
            **paired_kwargs,
            **base_kwargs
        )
    else:
        raise ValueError("Invalid argument: model_type '%s' not recognized." % model_type)
    if base_kwargs["is_train"]:
        model.init_optimizers()
        model.init_schedulers()
    return model
