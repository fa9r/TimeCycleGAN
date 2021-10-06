"""
Calculate average tOF score
original implementation: https://github.com/thunil/TecoGAN
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from timecyclegan.dataset.video_dataset import VideoDataset
from timecyclegan.models.networks.flownet2 import FlowNet2, flow_to_img
from timecyclegan.evaluation.eval_utils import split_tensor


def build_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dir0', help="First image directory.")
    parser.add_argument('dir1', help="Second image directory.")
    parser.add_argument(
        '--use_gpu', action='store_true', help='turn on flag to use GPU'
    )
    parser.add_argument(
        "--height", "-ih", type=int, default=64,
        help="Resize images to have image_height."
    )
    parser.add_argument(
        "--width", "-iw", type=int, default=64,
        help="Resize images to have image_width."
    )
    return parser


def calculate_tof(dir0, dir1, version=0.1, width=64, height=64):
    dataset = VideoDataset(
        root_dirs=[dir0, dir1],
        root_names=["dir0", "dir1"],
        block_size=2,
        overlap=True,
        augment_data=False,
        width=width,
        height=height,
    )
    dataloader = DataLoader(dataset)
    device = torch.device('cuda:0')
    model = FlowNet2(device=device)

    dist = 0
    count = 0
    for i, batch in enumerate(dataloader):
        if (i+1) % 100 == 0:
            print("Processing image [", i+1, "/", len(dataloader), "]")
        img0 = split_tensor(batch["dir0"].to(device))
        img1 = split_tensor(batch["dir1"].to(device))
        flow0 = model(*img0)[0].detach().cpu().numpy()
        flow1 = model(*img1)[0].detach().cpu().numpy()
        dist += np.sum(np.abs(flow0 - flow1))
        count += 1
    return dist / count


def main():
    parser = build_argparser()
    args = parser.parse_args()
    tof = calculate_tof(
        dir0=args.dir0,
        dir1=args.dir1,
        width=args.width,
        height=args.height,
    )
    print("tOF:", tof)


if __name__ == '__main__':
    main()
