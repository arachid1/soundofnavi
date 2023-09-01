from .CutMixSpecAugmenter import CutMixSpecAugmenter
from .MixUpSpecAugmenter import MixUpSpecAugmenter
from .AugMixAudioAugmenter import AugMixAudioAugmenter

def return_spec_augmenter(samples, spec_key, params):
    if spec_key == "cutmix":
        return CutMixSpecAugmenter(samples, **params)
    elif spec_key == "mixup":
        return MixUpSpecAugmenter(samples, **params)

def return_audio_augmenter(samples, audio_key, params):
    if audio_key == "augmix":
        return AugMixAudioAugmenter(samples, **params)
    elif audio_key == "ts":
        return TsAudioAugmenter(samples, **params)