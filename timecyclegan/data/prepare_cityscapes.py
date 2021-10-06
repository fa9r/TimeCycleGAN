"""
Script to prepare Cityscapes leftImg8bit_sequence_trainvaltest.zip for training
Originally, Cityscapes is organized as: leftimg8bit/<train|val|test>/<city>/<city>_<sequence>_<frame>_leftimg8bit.png.
This scripts rearranges that to: leftimg8bit/<train|val|test>/<city>_<sequence>/<frame>.png.
However, sequences in the original dataset are not consistent and sometimes contain multiple snippets.
Therefore, we also split and rename these sequences to be consistent with the rest.
In the end, there will be a separate folder for each snippet, and all of them are direct subfolders of <train|val|test>.
"""

import os
import argparse
from shutil import copyfile, move

from timecyclegan.util.os_utils import make_dir


INCONSISTENT_SEQUENCES = {
    'train': ['bochum', 'hanover', 'hamburg', 'krefeld', 'monchengladbach', 'strasbourg'],
    'val': ['frankfurt'],
    'test': ['bielefeld', 'mainz']
}


def process_frame(city_dir, image_name, output_dir, train_val_test, city, sequence, frame, copy=False):
    """
    Process frame city_dir/image_name and save it to output_dir/train_val_test/city_sequence/frame.png
    :param city_dir: Input directory of the current city (/.../cityscapes/.../<city>)
    :param image_name: Filename of the current frame
    :param output_dir: Output directory
    :param train_val_test: ('train' | 'val' | 'test')
    :param city: Name of the current city
    :param sequence: ID of the current city (6 digits, with leading 0s)
    :param frame: ID of the current frame in the sequence (6 digits, with leading 0s)
    :param copy: whether to copy or move the images
    """
    image_output_dir = os.path.join(output_dir, train_val_test, city + "_" + str(sequence))
    image_output_path = make_dir(os.path.join(image_output_dir, str(frame) + ".png"))
    move_or_copy = copyfile if copy else move
    move_or_copy(os.path.join(city_dir, image_name), image_output_path)


def build_arg_parser():
    """
    builds an argparser to run the application from command line
    :return: argparser
    """
    parser = argparse.ArgumentParser(description="Resize all images in a given directory.")
    parser.add_argument(
        "cityscapes_leftimg8bit_root",
        help="Path to cityscapes/leftimg8bit_sequence_trainvaltest/leftimg8bit_sequence/"
    )
    parser.add_argument("output_dir", help="Directory to which the results will be saved.")
    parser.add_argument(
        "--snippet_length", "-l", default=30, type=int,
        help="Length of the snippets. Default is 30, change to 1 to run/test on non-sequence dataset.")
    parser.add_argument("--copy", "-c", action='store_true', help="If set, copy the dataset instead of moving files.")
    return parser


def main():
    """main function"""
    parser = build_arg_parser()
    args = parser.parse_args()

    for train_val_test in os.listdir(args.cityscapes_leftimg8bit_root):
        train_val_test_dir = os.path.join(args.cityscapes_leftimg8bit_root, train_val_test)
        for city in sorted(os.listdir(train_val_test_dir)):  # here we sort to have more informative prints
            print("Processing", city, "in", train_val_test, "...")
            city_dir = os.path.join(train_val_test_dir, city)

            # for cities that have inconsistent sequence naming, we split and rename them manually
            if city in INCONSISTENT_SEQUENCES[train_val_test]:
                sequence = 0
                frame = 0
                for image_name in sorted(os.listdir(city_dir)):  # here we need to sort to have snippets together!
                    process_frame(
                        city_dir, image_name, args.output_dir, train_val_test, city, '%06d' % sequence, '%06d' % frame,
                        args.copy
                    )
                    if frame == args.snippet_length - 1:
                        sequence += 1
                        frame = 0
                    else:
                        frame += 1

            # for all other cities, we simply use the existing sequence/frame names
            else:
                for image_name in os.listdir(city_dir):  # here we do not sort to speed up processing time
                    sequence, frame = image_name.split("_")[1:3]
                    process_frame(
                        city_dir, image_name, args.output_dir, train_val_test, city, sequence, frame, args.copy
                    )


if __name__ == '__main__':
    main()
