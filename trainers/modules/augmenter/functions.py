from ..main import parameters

# from ..main.global_helpers import pad_sample
import nlpaug.augmenter.audio as naa
import librosa
import random


def shift_pitch(audio):
    aug = naa.PitchAug(sampling_rate=parameters.sr)
    return aug.augment(audio)


def stretch_time(audio, minval=0.6, maxval=1.2):
    factor = random.uniform(minval, maxval)
    audio = librosa.effects.time_stretch(audio, factor)
    desired_length = int(parameters.sr * parameters.audio_length)
    if len(audio) >= desired_length:
        # print("higher length: {}".format(len(audio)))
        audio = audio[:desired_length]
    else:
        # print("lower length: {}".format(len(audio)))
        audio = pad_sample(audio, desired_length)
    return audio
