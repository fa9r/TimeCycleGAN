"""
Inference code for frozen DeepLabV3 pretrained on Cityscapes images
Adjusted from https://github.com/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
Download frozen models from https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
CLass IDs: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
"""

import os
import tarfile
import argparse

from matplotlib import image as matimg
import numpy as np
from PIL import Image
import tensorflow as tf

from timecyclegan.util.os_utils import make_dir
from .colormaps import create_cityscapes_label_colormap


class DeepLabModel():
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 512
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """
        Creates and loads pretrained deeplab model.
        :param tarball_path: Path to the model tarball (download from tensorflow/models/research/deeplab github).
        """
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """
        Runs inference on a single image.
        :param image: A PIL.Image object, raw input image.
        :return:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def label_to_color_image(label):
    """
    Adds color defined by the dataset colormap to the label.
    :param label: A 2D array with integer type, storing the segmentation label.
    :return: A 2D array with floating type. The element of the array is the color indexed by the corresponding element
            in the input label to the PASCAL color map.
    :raises: ValueError: If label is not of rank 2 or its value is larger than color map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_cityscapes_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def run_model_on_image(model, image_path, output_path):
    """
    Inferences DeepLab model and saves result.
    :param model: loaded deeplab tensorflow model
    :param image_path: path from which to load the input image
    :param output_path: path to which the semantic segmentation result is saved
    :return:
    """
    original_img = Image.open(image_path)
    resized_img, seg_map = model.run(original_img)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    matimg.imsave(output_path, seg_image)


def build_arg_parser():
    """
    builds an argparser to run the application from command line
    :return: argparser
    """
    parser = argparse.ArgumentParser(description="Inference of frozen DeepLabV3 pretrained on Cityscapes images")
    parser.add_argument("model_path", help="Path to pretrained DeepLab model tarball (deeplab_cityscapes_....tar.gz).")
    parser.add_argument("image_dir", help="Path to images on which the deeplab model should be applied.")
    parser.add_argument("output_dir", help="Directory to which the semantic segmentation results are saved.")
    parser.add_argument(
        "--recursive", "-r", action='store_true',
        help="If set, runs deeplab on all sub-directories of image_dir instead."
    )
    return parser


def main():
    """Main function"""
    parser = build_arg_parser()
    args = parser.parse_args()

    # define deeplab model
    deeplab_model = DeepLabModel(args.model_path)

    # define image and output dirs; [""] will result in image_dirs=[args.image_dir] and output_dirs=[args.output_dir]
    sub_dirs = sorted(os.listdir(args.image_dir) if args.recursive else [""])
    image_dirs = [os.path.join(args.image_dir, sub_dir) for sub_dir in sub_dirs]
    output_dirs = [os.path.join(args.output_dir, sub_dir) for sub_dir in sub_dirs]

    # run deeplab on all images in all sub-directories
    for i, sub_dir in enumerate(sub_dirs):
        make_dir(output_dirs[i])
        print("----- Processing", sub_dir, ":-----")
        image_names = sorted(os.listdir(image_dirs[i]))
        for j, image_name in enumerate(image_names):
            print("Processing image", image_name, " [", j+1, "/", len(image_names), "]")
            image_path = os.path.join(image_dirs[i], image_name)
            output_path = os.path.join(output_dirs[i], image_name)
            run_model_on_image(deeplab_model, image_path, output_path)
        print("------------------------------------")


if __name__ == '__main__':
    main()
