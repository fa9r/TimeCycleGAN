"""Define argument parser"""

import argparse


class ArgumentParser:
    """This is a wrapper for argparse.ArgumentParser, which also defines all args"""

    def __init__(self):
        """Define base args used in all modes"""
        parser = argparse.ArgumentParser(description="TimeCycleGAN")
        # required args
        parser.add_argument(
            "model_name",
            help="Name of the model/experiment.\n"
                 "This will determine where checkpoints are saved (checkpoints/model_name)\n"
                 "and where training losses/images will be logged (runs/model_name)."
        )
        # optional args
        parser.add_argument(
            "--block_size", "-b", default=0, type=int,
            help="Load images in blocks of size block_size.\n"
                 "Default: 1 for basic model and cyclegan, 3 for recyclegan, 5 for block, timecycle, vid2vid models."
        )
        parser.add_argument(
            "--generator_architecture", "-g", default="resnet_6blocks",
            help="architecture of the generator ('resnet_6blocks' | 'resnet_9blocks' | 'unet_128' | 'unet_256')"
        )
        parser.add_argument(
            "--generator_filters", "-gf", type=int, default=64,
            help="Number of filters in the last conv layer of the generator."
        )
        parser.add_argument(
            "--n_frames_generator", "-gn", type=int, default=1,
            help="On how many previous generations the generator is conditioned."
        )
        parser.add_argument(
            "--gpu", "-gpu", type=int, default=-1, help="Which GPU to use for training (default: -1 -> CPU)."
        )
        parser.add_argument("--image_height", "-ih", type=int, default=256, help="Resize images to have image_height.")
        parser.add_argument("--image_width", "-iw", type=int, default=256, help="Resize images to have image_width.")
        parser.add_argument(
            "--model_type", "-m", default="basic",
            help="Which model to use ('cyclegan' | 'dcgan' | 'pix2pix' | 'recyclegan' | 'timecycle' | 'vid2vid').\n"
                 "'cyclegan': Unpaired CycleGAN model.\n"
                 "'dcgan': Unconditional DCGAN model.\n"
                 "'pix2pix': Paired Pix2Pix model.\n"
                 "'recyclegan' Unpaired RecycleGAN model.\n"
                 "'timecycle': Paired timecycle model.\n"
                 "'vid2vid': Paired Vid2Vid model."
        )
        parser.add_argument("--threads", "-t", type=int, default=1, help="Number of threads for data loading.")
        self.parser = parser

    def add_train_arguments(self):
        """Add train-specific args to the argparser"""
        parser = self.parser
        parser.add_argument("source_dir", help="Directory containing test source images.")
        parser.add_argument("target_dir", help="Directory containing test target images.")
        parser.add_argument("--batch_size", "-bs", default=1, type=int, help="Batch size.")
        parser.add_argument("--cycle_loss_weight", "-clw", default=0, type=int, help="Cycle loss weight.")
        parser.add_argument(
            "--discriminator_architecture", "-d", default="basic",
            help="architecture of the discriminator ('basic' | 'N_layers')"
        )
        parser.add_argument(
            "--discriminator_filters", "-df", type=int, default=64,
            help="Number of filters in the last conv layer of the discriminator."
        )
        parser.add_argument(
            "--n_frames_discriminator", "-dn", type=int, default=0,
            help="Number of frames the sequence discriminators discriminate."
        )
        parser.add_argument(
            "--discriminator_temporal_scales", "-dts", type=int, default=1,
            help="Number of temporal scales in framerate sampling (= number of sequence discriminators)."
        )
        parser.add_argument(
            "--feature_matching_loss_weight", "-fmlw", default=0, type=int, help="Loss weight of feature matching."
        )
        parser.add_argument(
            "--flow_loss_weight", "-flw", default=0, type=int, help="Loss weight of flow loss in vid2vid."
        )
        parser.add_argument(
            "--gan_mode", "-gan", default="lsgan", help="type of the gan loss ('vanilla' | 'lsgan' | 'wgangp')."
        )
        parser.add_argument(
            "--init_epoch", "-ie", default=0, type=int, help="If set, load models saved at a specific epoch."
        )
        parser.add_argument(
            "--init_checkpoint_dir", "-i",
            help="If set, initialize models from saved checkpoints in init_checkpoint_dir."
        )
        parser.add_argument(
            "--log_every", "-le", default=100, type=int, help="Log losses and images every log_every iterations."
        )
        parser.add_argument(
            "--log_images_every", "-lie", default=0, type=int,
            help="If specified, log images every log_images_every iterations, instead of every log_every iterations."
        )
        parser.add_argument("--load_height", "-lh", type=int, default=0, help="image load height (before cropping).")
        parser.add_argument("--l1_loss_weight", "-llw", default=0, type=int, help="L1 loss weight.")
        parser.add_argument("--learning_rate", "-lr", default=0.0002, type=float, help="Learning rate.")
        parser.add_argument("--load_width", "-lw", type=int, default=0, help="Image load width (before cropping).")
        parser.add_argument(
            "--mask_loss_weight", "-mlw", default=0, type=int, help="Loss weight of mask loss (weight loss) in vid2vid."
        )
        parser.add_argument("--num_epochs", "-ne", default=10, type=int, help="Number of training epochs.")
        parser.add_argument(
            "--perceptual_loss_weight", "-plw", default=0, type=int, help="Loss weight of perceptual (VGG19) loss."
        )
        parser.add_argument(
            "--recycle_loss_weight", "-rclw", default=0, type=int, help="Loss weight of recycle in RecycleGAN."
        )
        parser.add_argument(
            "--recycle_predictor_architecture", "-rcp", default="resnet_6blocks",
            help="Architecture of RecycleGAN predictor. See generator_architecture for options."
        )
        parser.add_argument(
            "--recycle_predictor_filters", "-rcpf", type=int, default=64,
            help="Number of filters in the last conv layer of the RecycleGAN predictor."
        )
        parser.add_argument(
            "--save_every", "-se", default=1, type=int, help="Save model checkpoints every save_every epochs."
        )
        parser.add_argument(
            "--spatial_scaling", "-ss", default=[1], type=float, nargs='+',
            help="Set steps for spatial scaling.\n"
                 "I.e. [0.25, 0.5, 1] to train a model with width and height 256 on 64 > 128 > 256 images."
        )
        parser.add_argument("--timecycle_loss", "-tcl", default="l1", help="Timecycle loss ('l1' | 'l2')")
        parser.add_argument("--timecycle_loss_weight", "-tclw", default=0, type=int, help="Timecycle loss weight.")
        parser.add_argument(
            "--timecycle_motion_model_architecture", "-tcmm", default="resnet_1blocks",
            help="Architecture of Timecycle motion model. See generator_architecture for options."
        )
        parser.add_argument(
            "--timecycle_motion_model_filters", "-tcmmf", type=int, default=64,
            help="Number of filters in the last conv layer of the Timecycle motion model."
        )
        parser.add_argument(
            "--timecycle_separate_motion_models", "-tcsmm", action="store_true",
            help="Set to use separate motion models for forward/backward predictions."
        )
        parser.add_argument(
            "--timecycle_type", "-tct", default="conditional",
            help="Type of Timecycle ('conditional' | 'pingpong')."
        )
        parser.add_argument(
            "--timecycle_warp_loss_weight", "-tcwlw", default=0, type=int, help="Timecycle warp loss weight."
        )
        parser.add_argument(
            "--temporal_scaling", "-ts", default=[1], type=float, nargs='+',
            help="Set steps for temporal scaling.\n"
                 "I.e. [0.2, 0.6, 1] to train a model with block_size 5 on 1 -> 3 -> 5 frames."
        )
        parser.add_argument(
            "--warp_loss_weight", "-wlw", default=0, type=int, help="Loss weight of warp loss in vid2vid."
        )

    def add_test_arguments(self):
        """Add test-specific args to the argparser"""
        parser = self.parser
        parser.add_argument(
            "test_source_dir", help="Directory of validation source images."
        )
        parser.add_argument(
            "test_target_dir", help="Directory of validation target images."
        )
        parser.add_argument("--test_batch_size", "-tbs", default=1, type=int, help="Test batch size.")
        parser.add_argument(
            "--test_init_epoch", "-tie", default=0, type=int, help="If set, load models saved at a specific epoch."
        )
        parser.add_argument("--test_output_dir", "-to", default="results", help="Directory to which results are written.")
        parser.add_argument(
            "--test_unpaired_target_to_source", "-tts", action="store_true",
            help="By default, our CycleGAN only creates target images from source images at test time. "
                 "If this flag is set, we create source from target images instead."
        )

    def add_val_arguments(self):
        """Add validation-specific args to the argparser"""
        self.add_test_arguments()

    def add_train_val_arguments(self):
        """Add args for both training and validation"""
        self.add_train_arguments()
        self.add_val_arguments()

    def parse_args(self):
        """Parse args"""
        return self.parser.parse_args()


def parse_args(mode=None):
    """
    Build an argparser for our application and parse its args.
    The argparser has different args depending on whether it is called by train.py or test.py.
    :param mode: Which function called the argparser (train | trainval | test)
    :return dict of args
    """
    parser = ArgumentParser()
    if mode == "train":
        parser.add_train_arguments()
    elif mode == "trainval":
        parser.add_train_val_arguments()
    elif mode == "val":
        parser.add_val_arguments()
    elif mode == "test":
        parser.add_test_arguments()
    else:
        raise ValueError(
            "build_argparser received incorrect mode."
            " Possible modes: ('train', 'trainval', 'val', 'test')."
        )
    return vars(parser.parse_args())
