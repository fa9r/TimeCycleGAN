"""
Flownet2 model implementation to be used to obtain 'ground truth' flows
Taken (and slightly adjusted) from https://github.com/NVlabs/few-shot-vid2vid
"""

import torch
from torch import nn
import torch.nn.functional as F

from .flownet2_pytorch import models as flownet2_models
from .flownet2_pytorch.utils import tools as flownet2_tools
from .flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d


class FlowNet2(nn.Module):
    def name(self):
        return 'FlowNet'

    def __init__(self, device):
        nn.Module.__init__(self)
        self.flowNet = flownet2_tools.module_to_dict(flownet2_models)['FlowNet2']().to(device)
        checkpoint = torch.load('timecyclegan/models/networks/flownet2_pytorch/FlowNet2_checkpoint.pth.tar')
        self.flowNet.load_state_dict(checkpoint['state_dict'])
        self.flowNet.eval()
        self.resample = Resample2d()
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    """ 
    def forward(self, data_list, epoch=0, dummy_bs=0):
        image_now, image_ref = data_list
        image_now, image_ref = image_now[:, :, :3], image_ref[:, 0:1, :3]
        flow_gt_prev = flow_gt_ref = conf_gt_prev = conf_gt_ref = None
        with torch.no_grad():
            if not self.opt.isTrain or epoch > self.opt.niter_single:
                image_prev = torch.cat([image_now[:, 0:1], image_now[:, :-1]], dim=1)
                flow_gt_prev, conf_gt_prev = self.flowNet_forward(image_now, image_prev)
            if self.opt.warp_ref:
                flow_gt_ref, conf_gt_ref = self.flowNet_forward(image_now, image_ref.expand_as(image_now))
            flow_gt, conf_gt = [flow_gt_ref, flow_gt_prev], [conf_gt_ref, conf_gt_prev]
            return flow_gt, conf_gt
    """

    def forward(self, input_A, input_B):
        with torch.no_grad():
            size = input_A.size()
            assert (len(size) == 4 or len(size) == 5)
            if len(size) == 5:
                b, n, c, h, w = size
                input_A = input_A.contiguous().view(-1, c, h, w)
                input_B = input_B.contiguous().view(-1, c, h, w)
                flow, conf = self.compute_flow_and_conf(input_A, input_B)
                return flow.view(b, n, 2, h, w), conf.view(b, n, 1, h, w)
            else:
                return self.compute_flow_and_conf(input_A, input_B)

    def compute_flow_and_conf(self, im1, im2):
        assert (im1.size()[1] == 3)
        assert (im1.size() == im2.size())
        old_size = im1.size()[2:4]
        scale_factor, new_size = None, None
        if old_size[0] < 256 or old_size[1] < 256:
            scale_factor = 256 / min(old_size[0], old_size[1])
        else:
            new_size = (val // 64 * 64 for val in old_size)
        if scale_factor is not None or old_size != new_size:
            im1 = F.interpolate(im1, size=new_size, scale_factor=scale_factor, mode='bilinear')
            im2 = F.interpolate(im2, size=new_size, scale_factor=scale_factor, mode='bilinear')
        self.flowNet.cuda(im1.get_device())
        data1 = torch.cat([im1.unsqueeze(2), im2.unsqueeze(2)], dim=2)
        flow1 = self.flowNet(data1)
        conf = (self.norm(im1 - self.resample(im2, flow1)) < 0.02).float()
        if scale_factor is not None or old_size != new_size:
            flow1 = F.interpolate(flow1, size=old_size, mode='bilinear') / (scale_factor or new_size[0] / old_size[0])
            conf = F.interpolate(conf, size=old_size, mode='bilinear')
        return flow1, conf

    def norm(self, t):
        return torch.sum(t * t, dim=1, keepdim=True)


def flow_warp(image, flow):
    """
    Optical flow warping: Warp given input image according to given optical flow
    Adjusted from few-shot-vid2vid models/networks/flownet.py resample(), get_grid()
    :param image: image to be warped
    :param flow: optical flow
    :return: flow-warped image
    """
    def get_grid(batchsize, rows, cols):
        hor = torch.linspace(-1.0, 1.0, cols)
        hor.requires_grad = False
        hor = hor.view(1, 1, 1, cols)
        hor = hor.expand(batchsize, 1, rows, cols)
        ver = torch.linspace(-1.0, 1.0, rows)
        ver.requires_grad = False
        ver = ver.view(1, 1, rows, 1)
        ver = ver.expand(batchsize, 1, rows, cols)
        t_grid = torch.cat([hor, ver], 1)
        t_grid.requires_grad = False
        return t_grid.cuda()
    b, c, h, w = image.size()
    grid = get_grid(b, h, w)
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
    final_grid = (grid + flow).permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(image, final_grid, mode='bilinear', padding_mode='border')
    return output


def flow_to_img(flows, height=64, width=64):
    """
    Transform optical flow to an image that can be logged to Tensorboard
    Adjusted from vid2vid/util/util.py tensor2flow(), which took it from here:
    https://stackoverflow.com/questions/28898346/visualize-optical-flow-with-color-model
    """
    import numpy as np
    import cv2
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    if isinstance(flows, list):
        flows = torch.cat(flows, 1)
    flows = flows.detach().view(-1, 2, height, width)
    num_flows = flows.shape[0]
    flows = flows.cpu().float().numpy()
    flow_images = []
    for i in range(num_flows):  # iterate over all flows
        flow = np.transpose(flows[i], (1, 2, 0))  # C x H x W -> H x W x C
        # construct HSV image
        flow_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_image[..., 0] = ang * 180 / np.pi / 2
        flow_image[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # convert HSV > RGB
        flow_image = cv2.cvtColor(flow_image, cv2.COLOR_HSV2RGB)
        # numpy > torch
        flow_images.append(to_tensor(flow_image))
    return flow_images
