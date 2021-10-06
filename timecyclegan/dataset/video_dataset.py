"""
Custom dataset to load videos (image sequences).
VideoDataset is a modified version of the AlignedDataset in Pix2Pix.
In contrast to AlignedDataset, VideoDataset also allows loading of image blocks (n subsequent image pairs at a time).
Data augmentation is also adjusted accordingly.
Furthermore, we made various minor code design changes.
Most notably, we don't use opt (and a long line of keyword args instead) and don't inherit from BaseDataset.
"""

import os
import random

import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


def read_image(image_path):
    """
    Reads an image from file as PIL.Image in RGB format
    :param image_path: path of the image file
    :return: PIL RBG image
    """
    return Image.open(image_path).convert('RGB')


def image_to_numpy(image):
    """
    Transforms a PIL.Image to a numpy array
    :param image: PIL image
    :return: numpy array
    """
    return np.asarray(image, dtype="float32") / 255


def get_lowest_level_dirs(root_dir):
    """
    For a given root directory, get a list of all subdirectories (including root_dir) which contain files
    :param root_dir: root directory
    :return: list of subdirectories containing files
    """
    file_paths = []
    for root, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            file_paths.append(os.path.join(root, file_name))
    return sorted(list({os.path.dirname(file_path) for file_path in file_paths}))


class VideoDataset(data.Dataset):
    """Custom dataset class to load images block-wise"""

    def __init__(
            self, root_dirs, root_names, block_size=1,
            overlap=False, augment_data=True,
            width=256, height=256, load_width=286, load_height=286,
    ):
        """
        Initialize the dataset
        :param root_dirs: List of directories containing the images
        :param root_names: Name of each dir in root_dirs
        :param block_size: Block size (how many images to process at once)
        :param overlap: Whether subsequent blocks should overlap or not
        :param augment_data: Whether to perform data augmentation (random crop + flip) or not
        :param width: Width of the loaded images (= random crop width if augment_data=True)
        :param height: Height of the loaded images (= random crop height if augment_data=True)
        :param load_width: Width to which each image is resized before the random crop (if augment_data=True)
        :param load_height: Height to which each image is resized before the random crop (if augment_data=True)
        """

        assert len(root_names) == len(root_dirs)  # make sure each root dir has a name

        self.root_names = root_names
        self.block_size = block_size
        self.overlap = overlap
        self.augment_data = augment_data
        self.width = width
        self.height = height
        self.load_width = load_width
        self.load_height = load_height

        # define data transformations applied similarly to all blocks: NHWC to NCHW > normalize
        transform_list_numpy = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.data_transform_numpy = transforms.Compose(transform_list_numpy)

        # define list of all sequence dirs for both source and target
        self.image_dirs = [get_lowest_level_dirs(root_dir) for root_dir in root_dirs]

        # define list of list of all image names for both source and target
        self.image_names = [[[f for f in sorted(os.listdir(image_dir)) if os.path.isfile(os.path.join(image_dir, f))] for image_dir in i] for i in self.image_dirs]

        # make sure all dirs contain same number of videos and frames
        for i in range(1, len(self.image_dirs)):
            assert len(self.image_dirs[0]) == len(self.image_dirs[i])
            for j in range(len(self.image_dirs[0])):
                assert len(self.image_names[0][j]) == len(self.image_names[i][j])

        # define lengths of each sequence (how many blocks can be loaded from each sequence)
        def get_sequence_len(sequence):
            """Return the length of a given sequence (i.e. the number of image blocks that can be loaded)"""
            if self.overlap:
                return max(0, len(sequence) - self.block_size + 1)
            return len(sequence) // self.block_size
        self.sequence_lens = [get_sequence_len(sequence) for sequence in self.image_names[0]]

        # get number of channels from first image for each directory
        self.image_channels = [
            image_to_numpy(read_image(os.path.join(self.image_dirs[i][0], self.image_names[i][0][0]))).shape[2]
            for i in range(len(self.image_dirs))
        ]

        # print dataset information
        self.print_info()

    def __len__(self):
        """
        Return the length of the dataset
        This is the sum of image blocks that can be loaded per sequence; all remaining frames will be discarded
        """
        return max(0, np.sum(self.sequence_lens))  # max(0,...) should not be needed, but pylint complains, so whatever

    def __getitem__(self, index):
        """
        Get an item at a specific index in the dataset
        Each item is a dict {source: x, target: y} where x is a source image block and y a target image block
        :param index: index of the item in the dataset
        :return: item at the specific index
        """

        # define block-specific data transformations: resize > random crop > random flip
        # This MUST be called here (and can not be done once in __init__()), because it is block-specific!
        data_transform = self.get_transforms()

        # initialize source and target image blocks, and list of source and target images names
        image_blocks = {
            root_name: np.zeros((self.block_size*channels, self.height, self.width), dtype="float32")
            for channels, root_name in zip(self.image_channels, self.root_names)
        }
        image_paths = {root_name+"_path": [] for root_name in self.root_names}

        # find sequence index (which sequence to load from) and block_index (which block to load in the sequence)
        sequence_index = 0
        sequence_len_sum = 0
        while sequence_len_sum + self.sequence_lens[sequence_index] < index + 1:
            sequence_len_sum += self.sequence_lens[sequence_index]
            sequence_index += 1
        block_index = index - sequence_len_sum

        # add images into source and target blocks
        for i in range(self.block_size):
            # find frame index (which frame to load in the sequence)
            image_index = block_index + i if self.overlap else block_index * self.block_size + i

            # load image and add it to the image block; add image path to image_paths
            for j, root_name in enumerate(self.root_names):
                image_name = self.image_names[j][sequence_index][image_index]
                image_path = os.path.join(self.image_dirs[j][sequence_index], image_name)
                image_paths[root_name+"_path"].append(image_path)
                image = self.read_and_transform(image_path, data_transform)
                image_blocks[root_name][i*self.image_channels[j]:(i+1)*self.image_channels[j], :, :] = image

        return {**image_blocks, **image_paths}

    def read_and_transform(self, image_path, data_transform):
        """
        Reads an image and performs several transformations
        :param image_path: path of the image file
        :param data_transform: block-specific data transformations
        :return: transformed image (np.array)
        """
        image = read_image(image_path)  # read image from image_path as Pillow.Image
        image = data_transform(image)  # perform block-specific data transformations
        image = image_to_numpy(image)  # convert to [0,1]-normalized numpy array
        image = self.data_transform_numpy(image)  # perform further data transformations
        return image

    def get_transforms(self):
        """
        Define block-specific data transformations: resize > random crop > random flip
        :return: data transformations (torchvision.transforms.Compose)
        """
        transform_list = []
        crop_size = (self.width, self.height)
        load_size = (self.load_width, self.load_height)
        if self.augment_data:
            crop_location, do_flip = self.sample_transformation_params(load_size, crop_size)
            transform_list.append(transforms.Resize(load_size, Image.BICUBIC))
            transform_list.append(
                transforms.Lambda(lambda img: self.parametrized_random_crop(img, crop_location, crop_size))
            )
            transform_list.append(transforms.Lambda(lambda img: self.parametrized_random_flip(img, do_flip)))
        else:
            transform_list.append(transforms.Resize(crop_size, Image.BICUBIC))
        return transforms.Compose(transform_list)

    def print_info(self):
        """Print dataset size, source dimensions and target dimensions"""
        print("---------- Dataset Information ----------")
        print("dataset size:", len(self))
        for i, root_name in enumerate(self.root_names):
            print(root_name, "dimensions:", self.image_channels[i], "x", self.width, "x", self.height)
        print("-----------------------------------------")

    @staticmethod
    def parametrized_random_crop(img, crop_position, crop_size):
        """
        Parametrized Random Crop
        :param img: image to perform random cropping on
        :param crop_position: start position of the random crop (top left corner) - (x, y)
        :param crop_size: random crop size (width, height)
        :return: random crop
        """
        image_width, image_height = img.size
        crop_x, crop_y = crop_position
        crop_width, crop_height = crop_size
        if image_width > crop_width or image_height > crop_height:
            return img.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
        return img

    @staticmethod
    def parametrized_random_flip(img, do_flip):
        """
        Parametrized Horizontal Random Flip
        :param img: image to perform random flip on
        :param do_flip: whether to flip the image or not
        :return: flipped image if do_flip else original input image
        """
        if do_flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def sample_transformation_params(load_size, crop_size):
        """
        Sample parameters for random crop (crop_postion) and for random flip (do_flip)
        :param load_size: size (width, height) of the image before the random crop
        :param crop_size: random crop size (width, height)
        :return: crop position, do_flip
        """
        load_width, load_height = load_size
        crop_width, crop_height = crop_size
        crop_x = random.randint(0, np.maximum(0, load_width - crop_width))
        crop_y = random.randint(0, np.maximum(0, load_height - crop_height))
        crop_position = (crop_x, crop_y)
        do_flip = random.random() > 0.5
        return crop_position, do_flip
