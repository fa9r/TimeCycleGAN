"""Timecycle Models"""

from timecyclegan.models.mixins.timecycle_mixin import TimeCycleMixinPaired
from timecyclegan.models.baselines.vid2vid_model import Vid2VidModel


class TimeCycleVid2Vid(TimeCycleMixinPaired, Vid2VidModel):
    """
    Vid2Vid + Timecycle
    Adds a paired timecycle to pix2pix, and runs all vid2vid discriminators on the generations of the timecycle
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_timecycle_discriminator(self.discriminator)
