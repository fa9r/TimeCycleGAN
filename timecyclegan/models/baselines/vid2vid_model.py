"""Reimplemented (and strongly simplified) vid2vid (https://github.com/NVIDIA/vid2vid/)"""

import torch

from timecyclegan.models.sequential import SequentialPix2Pix
from timecyclegan.models.networks.flownet2 import FlowNet2, flow_warp, flow_to_img


class Vid2VidModel(SequentialPix2Pix):
    """vid2vid model"""
    def __init__(
            self, flow_loss_weight=0, warp_loss_weight=0, mask_loss_weight=0,
            n_frames_D=3, temporal_scales=2, div_flow=20,
            use_modular_networks=True, use_masked_flow_loss=True, use_flow=True,
            log_intermediate_images=True, log_flows=True,
            **kwargs
    ):
        """
        Initialize the model
        :param feature_matching_loss_weight: relative weight of the feature matching loss
        :param perceptual_loss_weight: relative weight of the perceptual loss
        :param flow_loss_weight: relative weight of the flow loss
        :param n_frames_G: How many previous generations to condition G on
        :param n_frames_D: Number of frames to feed into temporal discriminator
        :param temporal_scales: Number of temporal scales in framerate sampling (= number of sequence discriminators)
        :param log_intermediate_images: Whether to log intermediate generations (before mask-combining) to Tensorboard
        :param use_modular_networks: If true, use separate models for target feat extr., source feat extr., ...
        :param div_flow: Multiply flow by div_flow to learn smaller values for improved stability (see flownet2-pytorch)
        :param use_masked_flow_loss: If set, calculate flow loss and warp loss with flownet confidence as mask
        :param use_flow: If False, do not use any flow warping in the image generation process
        :param kwargs: Additional keyword args for PairedModel
        """
        super().__init__(G_input_conditioning=True, **kwargs)
        self.n_frames_D = n_frames_D
        self.hparam_dict["Sequence Discriminator Input Length"] = n_frames_D
        self.temporal_scales = temporal_scales
        self.hparam_dict["Discriminator Temporal Scales"] = temporal_scales
        self.flow_loss_weight = flow_loss_weight
        self.hparam_dict["Flow Loss Weight"] = flow_loss_weight
        self.warp_loss_weight = warp_loss_weight
        self.hparam_dict["Warp Loss Weight"] = warp_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.hparam_dict["WeightMask Loss Weight"] = mask_loss_weight
        self.use_modular_networks = use_modular_networks
        self.hparam_dict["Modular Network Design"] = use_modular_networks
        self.div_flow = div_flow
        self.hparam_dict["Flow Multiplier"] = div_flow
        self.use_masked_flow_loss = use_masked_flow_loss
        self.hparam_dict["Masked Flow Loss"] = use_masked_flow_loss
        self.use_flow = use_flow
        self.hparam_dict["Flow"] = use_flow
        self.log_intermediate_images = log_intermediate_images
        self.log_flows = log_flows
        # define generators
        self.generators = {}
        if self.use_modular_networks:
            num_blocks = int(self.generator_architecture[7:-6])
            num_blocks_down = num_blocks // 2
            num_blocks_up = num_blocks - num_blocks_down
            architecture_down = "resnet_%dblocks" % num_blocks_down
            architecture_up = "resnet_%dblocks" % num_blocks_up
            self.feature_extractor_source = self.define_G(
                input_channels=self.source_channels * (self.n_frames_G_condition + 1),
                output_channels=-1,
                architecture=architecture_down,
                filters=self.generator_filters//2,
                part="down",
            )
            self.feature_extractor_target = self.define_G(
                input_channels=self.target_channels * self.n_frames_G,
                output_channels=-1,
                architecture=architecture_down,
                filters=self.generator_filters - self.generator_filters//2,  # source filters + target filters = gen filters
                part="down",
            )
            self.generator = self.define_G(
                input_channels=-1,
                output_channels=self.target_channels,
                architecture=architecture_up,
                output_activation="tanh",
                part="up_head"
            )
            self.flow_mask_net = self.define_G(
                input_channels=-1,
                output_channels=-1,
                architecture=architecture_up,
                part="up"
            )
            self.generators["feature_extractor_source"] = self.feature_extractor_source
            self.generators["feature_extractor_target"] = self.feature_extractor_target
        else:
            self.generator = self.define_G(
                input_channels=self.source_channels * (self.n_frames_G_condition + 1) + self.target_channels * self.n_frames_G,
                output_channels=self.target_channels,
                output_activation="tanh",
            )
            self.flow_mask_net = self.define_G(
                input_channels=self.source_channels * (self.n_frames_G_condition + 1) + self.target_channels * self.n_frames_G,
                output_channels=-1,
                part="down_up",
            )
        self.generators["generator"] = self.generator
        self.generators["flow_mask_net"] = self.flow_mask_net
        self.flow_head = self.define_G(
            input_channels=-1,
            output_channels=2,
            output_activation="linear",
            part="head",
        )
        self.mask_head = self.define_G(
            input_channels=-1,
            output_channels=1,
            output_activation="sigmoid",
            part="head",
        )
        self.generators["flow_head"] = self.flow_head
        self.generators["mask_head"] = self.mask_head
        # define discriminators, flownet, and initialize class args
        if self.is_train:
            self.discriminators_seq = []
            for i in range(temporal_scales):
                discr_seq = self.define_D(self.target_channels * n_frames_D + 2 * (n_frames_D - 1), store_layers=True)
                self.discriminators_seq.append(discr_seq)
                self.discriminators["discriminator_seq" + (("_" + str(i+1)) if i > 0 else "")] = discr_seq
            # define FlowNet2
            self.flownet = FlowNet2(self.device)
            # initialize ground truth flow; this will be set during training in set_input()
            self.real_flow, self.real_flow_conf = [None] * 2
            # initialize fakes that will be set during training in the forward pass in forward()
            self.fake_flow, self.fake_flow_mask, self.fake_target_flow, self.fake_target_gen = [None] * 4
            # initialize losses; these will be set during the backward pass when calling optimize_parameters()
            self.losses_G_GAN_seq, self.losses_G_fm_seq = [None] * 2
            self.loss_G_flow, self.loss_G_flow_warp, self.loss_G_flow_mask, self.loss_G_warp = [None] * 4
            self.losses_D_seq, self.losses_D_seq_real, self.losses_D_seq_fake = [None] * 3

    def set_input(self, input_data):
        super().set_input(input_data)
        if self.is_train:
            self.real_flow, self.real_flow_conf = [], []
            for i in range(max(1, self.temporal_scales)):
                step = self.n_frames_D ** i
                flows, confs = [], []
                for j in range((self.block_size - 1) // step):
                    flow, conf = self.flownet(self.real_target[(j+1)*step], self.real_target[j*step])
                    flows.append(flow)
                    confs.append(conf)
                self.real_flow.append(flows)
                self.real_flow_conf.append(confs)

    def generate_image(self, input_):
        """
        Generate an image according to the given generator input
        :return: Final generation, predicted optical flow, generated images by flow-warping,
            image generated by generator, predicted mask to combine generations
        """
        sources = input_[:, :self.source_channels * (self.n_frames_G_condition + 1)]
        targets = input_[:, self.source_channels * (self.n_frames_G_condition + 1):]
        assert targets.size()[1] == self.target_channels * self.n_frames_G
        # define generator input
        if self.use_modular_networks:
            source_latent = self.feature_extractor_source(sources)
            target_latent = self.feature_extractor_target(targets)
            generator_input = torch.cat((source_latent, target_latent), 1)
        else:
            generator_input = torch.cat((sources, targets), 1)
        # generate fakes
        fake_target_gen = self.generator(generator_input)  # generator fake
        flow_mask_latent = self.flow_mask_net(generator_input)
        fake_flow = self.flow_head(flow_mask_latent) * self.div_flow
        if self.use_flow:
            fake_target_flow = flow_warp(targets[:, -self.target_channels:], fake_flow)  # flow fake
            mask = self.mask_head(flow_mask_latent)
            fake_target = mask * fake_target_gen + (1 - mask) * fake_target_flow  # final fake
            if self.log_intermediate_images:
                return fake_target, fake_flow, mask, fake_target_flow, fake_target_gen
            return fake_target, fake_flow, mask, None, None
        return fake_target_gen, fake_flow, None, None, None

    def forward(self):
        fake_flows, fake_flow_masks, fake_targets_flow, fake_targets_gen = [], [], [], []
        fake_targets = self.get_placeholder_images(self.n_frames_G)
        sources = self.get_placeholder_images(self.n_frames_G, channels=self.source_channels)
        for i in range(self.block_size):
            sources.append(self.real_source[i])
            fakes = self.generate_image(torch.cat([*sources[i:], *fake_targets[i:]], 1))
            fake_target, fake_flow, fake_flow_mask, fake_target_flow, fake_target_gen = fakes
            fake_targets.append(fake_target)
            fake_flows.append(fake_flow)
            if self.use_flow:
                fake_flow_masks.append(fake_flow_mask)
                if self.log_intermediate_images:
                    fake_targets_flow.append(fake_target_flow.detach())
                    fake_targets_gen.append(fake_target_gen.detach())
        self.fake_target = fake_targets[self.n_frames_G:]
        self.fake_flow = fake_flows[1:]  # ignore first flow predicted from random noise
        if self.use_flow:
            self.fake_flow_mask = fake_flow_masks[1:]
            if self.log_intermediate_images:
                self.fake_target_flow = torch.cat(fake_targets_flow, 1)
                self.fake_target_gen = torch.cat(fake_targets_gen, 1)

    def backward_D_seq(self):
        """Backward pass for sequence discriminator"""
        self.losses_D_seq, self.losses_D_seq_real, self.losses_D_seq_fake = [], [], []
        for i, discriminator_seq in enumerate(self.discriminators_seq):
            loss_D_seq_real, loss_D_seq_fake = 0, 0
            step = self.n_frames_D ** i
            seq_len = (self.n_frames_D - 1) * step
            num_discr = (self.block_size - 1) // seq_len
            for j in range(num_discr):
                indices = [j*seq_len + k*step for k in range(self.n_frames_D)]
                flow_ind = j * (self.n_frames_D - 1)
                real_flow = torch.cat(self.real_flow[i][flow_ind:flow_ind+self.n_frames_D-1], 1) / self.div_flow
                real_target = torch.cat([self.real_target[ind] for ind in indices], 1)
                fake_target = torch.cat([self.fake_target[ind] for ind in indices], 1).detach()
                loss_D_seq_real += self.get_GAN_loss(real_target, discriminator_seq, True, real_flow)
                loss_D_seq_fake += self.get_GAN_loss(fake_target, discriminator_seq, False, real_flow)
            self.losses_D_seq_real.append((loss_D_seq_real / num_discr) if num_discr > 0 else 0)
            self.losses_D_seq_fake.append((loss_D_seq_fake / num_discr) if num_discr > 0 else 0)
            self.losses_D_seq.append((self.losses_D_seq_real[i] + self.losses_D_seq_fake[i]) / 2)

    def backward_D(self):
        """Calculate discriminator losses"""
        self.backward_D_seq()
        self.loss_D += sum(self.losses_D_seq)
        super().backward_D()

    def backward_G_seq(self):
        self.losses_G_GAN_seq, self.losses_G_fm_seq = [], []
        """Calculate sequence GAN and feature matching losses for the generators"""
        for i, discriminator_seq in enumerate(self.discriminators_seq):
            loss_G_GAN_seq, loss_G_fm_seq = 0, 0
            step = self.n_frames_D ** i
            seq_len = (self.n_frames_D - 1) * step
            num_discr = (self.block_size - 1) // seq_len
            for j in range(num_discr):
                indices = [j*seq_len + k*step for k in range(self.n_frames_D)]
                flow_ind = j * (self.n_frames_D - 1)
                real_flow = torch.cat(self.real_flow[i][flow_ind:flow_ind+self.n_frames_D-1], 1) / self.div_flow
                real_target = torch.cat([self.real_target[ind] for ind in indices], 1)
                fake_target = torch.cat([self.fake_target[ind] for ind in indices], 1)
                gan_loss, fm_loss = self.get_FM_loss(fake_target, real_target, discriminator_seq, real_flow)
                loss_G_GAN_seq += gan_loss
                loss_G_fm_seq += fm_loss
            self.losses_G_GAN_seq.append((loss_G_GAN_seq / num_discr) if num_discr > 0 else 0)
            self.losses_G_fm_seq.append((loss_G_fm_seq / num_discr) if num_discr > 0 else 0)

    def backward_G_flow(self):
        """Calculate flow and warp losses"""
        loss_G_flow, loss_G_flow_warp, loss_G_warp, loss_G_flow_mask = [0] * 4
        for i in range(self.block_size - 1):
            loss_mask = (self.real_flow_conf[0][i] if self.use_masked_flow_loss else 1)
            # flow loss
            loss_G_flow += self.criterion_L1(
                self.fake_flow[i] * loss_mask,
                self.real_flow[0][i] * loss_mask
            )
            loss_G_flow_warp += self.criterion_L1(
                flow_warp(self.real_target[i], self.fake_flow[i]) * loss_mask,
                self.real_target[i+1] * loss_mask
            )
            # warp loss
            loss_G_warp += self.criterion_L1(
                flow_warp(self.fake_target[i], self.real_flow[0][i]).detach() * loss_mask,
                self.fake_target[i+1] * loss_mask
            )
            if self.use_flow:
                # flow mask loss (vid2vid "weight loss")
                loss_G_flow_mask += self.criterion_L1(
                    self.fake_flow_mask[i] * loss_mask,
                    torch.zeros_like(self.fake_flow_mask[i]),
                )
        self.loss_G_flow = loss_G_flow / (self.block_size - 1)
        self.loss_G_flow_warp = loss_G_flow_warp / (self.block_size - 1)
        self.loss_G_warp = loss_G_warp / (self.block_size - 1)
        self.loss_G_flow_mask = loss_G_flow_mask / (self.block_size - 1)

    def backward_G(self):
        """Calculate generator losses"""
        self.backward_G_seq()
        self.backward_G_flow()
        # combine losses and backprop
        self.loss_G += sum(self.losses_G_GAN_seq)
        self.loss_G += self.feature_matching_loss_weight * sum(self.losses_G_fm_seq)
        self.loss_G += self.flow_loss_weight * (self.loss_G_flow + self.loss_G_flow_warp)
        self.loss_G += self.warp_loss_weight * self.loss_G_warp
        self.loss_G += self.mask_loss_weight * self.loss_G_flow_mask
        super().backward_G()

    def define_logging(self):
        super().define_logging()
        # define losses to be logged
        losses = {
            **{"Generator/GAN/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_G_GAN_seq)},
            **{"Generator/FeatureMatching/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_G_fm_seq)},
            "Generator/Flow/Flow": self.loss_G_flow,
            "Generator/Flow/Warp": self.loss_G_flow_warp,
            "Generator/WeightMask": self.loss_G_flow_mask,
            "Generator/Warp": self.loss_G_warp,
            **{"Discriminator/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_D_seq)},
            **{"Discriminator/Sequence/%d/Real" % i: loss for i, loss in enumerate(self.losses_D_seq_real)},
            **{"Discriminator/Sequence/%d/Fake" % i: loss for i, loss in enumerate(self.losses_D_seq_fake)},
        }
        self.losses = {**self.losses, **losses}
        # define additional images to be logged
        flow_image_data = {"nrow": self.block_size - 1, "val_range": (0, 1)}
        mask_image_data = {"channels": 1, "nrow": self.block_size - 1, "val_range": (0, 1)}
        if self.use_flow and self.log_intermediate_images:
            intermediate_images = {
                'Generations/Generator': self.fake_target_gen,
                'Generations/Flow': self.fake_target_flow,
                'Generations/Mask': (self.fake_flow_mask, mask_image_data)
            }
            self.images = {**self.images, **intermediate_images}
        if self.log_flows:
            flow_images = {
                'Flow/Fake': (flow_to_img(self.fake_flow, height=self.height, width=self.width), flow_image_data),
                'Flow/Real': (flow_to_img(self.real_flow[0], height=self.height, width=self.width), flow_image_data)
            }
            if self.use_masked_flow_loss:
                flow_images['Flow/Conf'] = (self.real_flow_conf[0], mask_image_data)
            self.images = {**self.images, **flow_images}

    def test(self):
        return self.test_sequential(
            generator=lambda x: self.generate_image(x)[0],
            fake_channels=self.target_channels,
            cond=self.real_source,
            cond_channels=self.source_channels,
        )
