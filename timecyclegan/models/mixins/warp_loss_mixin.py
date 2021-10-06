from timecyclegan.models.base_model import BaseModel


class WarpLossMixin(BaseModel):
    def __init__(self, warp_loss_weight, *args, **kwargs):
        from timecyclegan.models.networks.flownet2 import FlowNet2
        super().__init__(*args, **kwargs)
        self.warp_loss_weight = warp_loss_weight
        self.warp_flownet = FlowNet2(self.device)
        self.warp_seq_real, self.warp_seq_fake = None, None

    def set_warp_loss_sequences(self, real_sequence, fake_sequence):
        self.warp_seq_real = real_sequence
        self.warp_seq_fake = fake_sequence

    def backward_G_warp(self):
        from timecyclegan.models.networks.flownet2 import flow_warp
        # calculate flows on sequence
        flows, confs = [], []
        for i in range(self.block_size - 1):
            flow, conf = self.warp_flownet(
                self.warp_seq_real[i + 1],
                self.warp_seq_real[i])
            flows.append(flow)
            confs.append(conf)
        # calculate warp loss
        loss_G_warp = 0
        for i in range(self.block_size - 1):
            loss_mask = (confs[i])
            loss_G_warp += self.criterion_L1(
                flow_warp(
                    self.warp_seq_fake[i],
                    flows[i]
                ).detach() * loss_mask,
                self.warp_seq_fake[i + 1] * loss_mask
            )
        self.loss_G_warp = loss_G_warp / (self.block_size - 1)
        return self.loss_G_warp

    def backward_G(self):
        self.backward_G_warp()
        self.loss_G += self.warp_loss_weight * self.loss_G_warp
        super().backward_G()

    def define_logging(self):
        super().define_logging()
        self.losses = {**self.losses, "Generator/Warp": self.loss_G_warp}
