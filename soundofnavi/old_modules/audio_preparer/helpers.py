
from .IcbhiAudioPreparer import IcbhiAudioPreparer
from .JordanAudioPreparer import JordanAudioPreparer
from .BdAudioPreparer import BdAudioPreparer
from .PerchAudioPreparer import PerchAudioPreparer
from .AntwerpAudioPreparer import AntwerpAudioPreparer
from ..main import parameters

def return_preparer(key, samples):
    if key == "Icbhi":
        return IcbhiAudioPreparer(samples, key, mode=parameters.mode)
    elif key == "Jordan":
        return JordanAudioPreparer(samples, key, mode=parameters.mode)
    elif key == "Bd":
        return BdAudioPreparer(samples, key, mode=parameters.mode)
    elif key == "Perch":
        return PerchAudioPreparer(samples, key, mode=parameters.mode)
    elif key == "Antwerp":
        return AntwerpAudioPreparer(samples, key, mode=parameters.mode)
