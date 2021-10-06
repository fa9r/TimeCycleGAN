"""
Calculate average tLP score
original implementation: https://github.com/thunil/TecoGAN
"""

import argparse

import numpy as np
from torch.utils.data import DataLoader

from timecyclegan.dataset.video_dataset import VideoDataset
from timecyclegan.evaluation.PerceptualSimilarity.models import PerceptualLoss
from timecyclegan.evaluation.eval_utils import split_tensor

def build_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dir0', help="First image directory.")
    parser.add_argument('dir1', help="Second image directory.")
    parser.add_argument('-v','--version', type=str, default='0.1')
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


def calculate_tlp(dir0, dir1, version=0.1, use_gpu=False, width=64, height=64):
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
    model = PerceptualLoss(
        model='net-lin',
        net='alex',
        use_gpu=use_gpu,
        version=version,
    )

    dist = 0
    count = 0
    for i, batch in enumerate(dataloader):
        if (i+1) % 100 == 0:
            print("Processing image [", i+1, "/", len(dataloader), "]")
        img0 = split_tensor(batch["dir0"])
        img1 = split_tensor(batch["dir1"])
        if use_gpu:
            img0 = [tensor.cuda() for tensor in img0]
            img1 = [tensor.cuda() for tensor in img1]
        dist_0 = model.forward(*img0).item()
        dist_1 = model.forward(*img1).item()
        dist += np.abs(dist_0 - dist_1)
        count += 1
    return dist / count


def main():
    parser = build_argparser()
    args = parser.parse_args()
    tlp = calculate_tlp(
        dir0=args.dir0,
        dir1=args.dir1,
        version=args.version,
        use_gpu=args.use_gpu,
        width=args.width,
        height=args.height,
    )
    print("tLP:", tlp)


if __name__ == '__main__':
    main()
