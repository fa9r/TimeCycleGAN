"""Script to resize all images in a given directory"""

import os
import argparse

from PIL import Image

from timecyclegan.util.os_utils import make_dir


def resize_image(image_path, output_path, width, height):
    """
    Resize a given image
    :param image_path: Path to the image file
    :param output_path: Path to where the resized image should be saved
    :param width: Resize width
    :param height: Resize height
    """
    img = Image.open(image_path)
    img_resize = img.resize((width, height), Image.ANTIALIAS)
    img_resize.save(output_path, 'PNG')


def build_arg_parser():
    """
    builds an argparser to run the application from command line
    :return: argparser
    """
    parser = argparse.ArgumentParser(
        description="Resize all images in a given directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_dir", help="Path to directory containing the images to be resized.")
    parser.add_argument("output_dir", help="Directory to which the resized images will be saved.")
    parser.add_argument("--width", "-w", default=256, type=int, help="Resize width.")
    parser.add_argument("--height", "-x", default=256, type=int, help="Resize height.")
    parser.add_argument("--rename", "-r", action='store_true', help="Rename resized images to 1.png, 2.png, ....")
    parser.add_argument("--recursive", "-s", action='store_true', help="Rum on sub-directories instead.")
    return parser


def main():
    """main function"""
    parser = build_arg_parser()
    args = parser.parse_args()
    sub_dirs = sorted(os.listdir(args.image_dir) if args.recursive else [""])
    image_dirs = [os.path.join(args.image_dir, sub_dir) for sub_dir in sub_dirs]
    output_dirs = [os.path.join(args.output_dir, sub_dir) for sub_dir in sub_dirs]

    for i, sub_dir in enumerate(sub_dirs):
        make_dir(output_dirs[i])
        print("----- Processing", sub_dir, ":-----")
        image_files = os.listdir(image_dirs[i])
        for j, image_filename in enumerate(sorted(image_files)):
            print("Resizing image ", image_filename, " [", j + 1, "/", len(image_files), "].")
            image_path = os.path.join(image_dirs[i], image_filename)
            output_path = os.path.join(output_dirs[i], str(j+1) + ".png") if args.rename \
                else os.path.join(output_dirs[i], image_filename)
            resize_image(image_path, output_path, args.width, args.height)
        print("------------------------------------")



if __name__ == '__main__':
    main()
