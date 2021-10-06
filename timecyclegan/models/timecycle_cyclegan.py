from .mixins import TimecycleSeqDMixinUnpaired, TimeCycleMixinUnpaired, SeqDMixinUnpaired
from .sequential import SequentialCycleGAN


class TimeCycleCycleGAN(TimeCycleMixinUnpaired, SeqDMixinUnpaired, SequentialCycleGAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_timecycle_discriminator(self.discriminator_target, timecycle_name="target")
        self.add_timecycle_discriminator(self.discriminator_source, timecycle_name="source")
