"""
Script to combine files from all subfolders.
This is for instance useful to create a single video from all train images for visualization purposes.
The combination can be undone by running the script again with -u flag.
"""

import argparse
import os
from shutil import move

from timecyclegan.util.os_utils import make_dir


def build_arg_parser():
    """
    builds an argparser to run the application from command line
    :return: argparser
    """
    parser = argparse.ArgumentParser(description="Move all files from subdirectories to the current directory.")
    parser.add_argument(
        "--uncombine", "-u", action="store_true",
        help="Set this flag to undo a previous combination and move everything back to the subfolders."
    )
    return parser


def combine():
    """Combine subfolders, i.e. move all files from subfolders to current folder"""
    sub_dir_names = sorted(os.listdir(os.curdir))
    for sub_dir_name in sub_dir_names:
        sub_dir = os.path.join(os.curdir, sub_dir_name)
        for file_name in sorted(os.listdir(sub_dir)):
            file_path = os.path.join(sub_dir, file_name)
            move(file_path, os.path.join(os.curdir, sub_dir_name + "_" + file_name))


def uncombine():
    """Undo a previous combine(), i.e. move all files from current folder back to subfolders"""
    file_names = sorted([f for f in os.listdir(os.curdir) if os.path.isfile(os.path.join(os.curdir, f))])
    for file_name in file_names:

        # split filename into subdir and new filename
        name_parts = file_name.split("_")
        new_file_name = name_parts[-1]
        dir_name = "_".join(name_parts[:-1])
        dir_path = make_dir(os.path.join(os.curdir, dir_name))

        # move file to new path
        file_path = os.path.join(os.curdir, file_name)
        new_file_path = os.path.join(dir_path, new_file_name)
        move(file_path, new_file_path)


def main():
    """Main function"""
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.uncombine:
        uncombine()
    else:
        combine()


if __name__ == '__main__':
    main()
