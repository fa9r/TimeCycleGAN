"""
Modified Pix2Pix model to also learn temporal consistency of subsequent frames in video generation.
To learn temporal consistency, we add a cycle loss (from CycleGAN) between subsequent frames.
This is achieved by having three generators, one that predicts frames from semantic segmentation maps, one that
    predicts frame t from frame t-1, and one that predicts frame t-1 from frame t.
"""

from timecyclegan.models.mixins import (
    TimeCycleMixin,
    TimeCycleMixinPaired,
    SeqDMixinPaired,
    TimecycleSeqDMixinPaired,
    WarpLossMixin,
)

from .sequential import SequentialPix2Pix


class TimeCyclePix2Pix(TimeCycleMixinPaired, SeqDMixinPaired, SequentialPix2Pix):
    def __init__(self, timecycle_type="conditional", **kwargs):
        super().__init__(timecycle_type=timecycle_type, **kwargs)
        if timecycle_type == "pingpong":
            self.define_timecycle_pingpong(self.generator)
        self.add_timecycle_discriminator(self.discriminator, 1)


class TimeCyclePix2PixWarp(WarpLossMixin, TimeCyclePix2Pix):
    def forward(self):
        super().forward()
        self.set_warp_loss_sequences(
            real_sequence=self.real_target,
            fake_sequence=self.fake_target,
        )
