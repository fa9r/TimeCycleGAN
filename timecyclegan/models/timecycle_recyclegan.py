from timecyclegan.models.mixins import TimecycleSeqDMixinUnpaired, TimeCycleMixinUnpaired, SeqDMixinUnpaired
from timecyclegan.models.baselines.recyclegan_model import RecycleGANModel
from .sequential import SequentialRecycleGAN


class TimeCycleRecycleGAN(TimecycleSeqDMixinUnpaired, RecycleGANModel):
    """
    RecycleGAN + Sequence Discriminator + Timecycle
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_timecycle_discriminator(self.discriminator_target, timecycle_name="target")
        self.add_timecycle_discriminator(self.discriminator_source, timecycle_name="source")


class TimeCycleRecycleGANv2(TimeCycleMixinUnpaired, SeqDMixinUnpaired, SequentialRecycleGAN):
    def __init__(self, timecycle_type="unconditional", **kwargs):
        assert timecycle_type == "unconditional"
        super().__init__(timecycle_type="timerecycle", **kwargs)
        self.define_timecycle_timerecyclev2(self.predictor_target, timecycle_name="target")
        self.define_timecycle_timerecyclev2(self.predictor_source, timecycle_name="source")
        self.add_timecycle_discriminator(self.discriminator_target, timecycle_name="target")
        self.add_timecycle_discriminator(self.discriminator_source, timecycle_name="source")
