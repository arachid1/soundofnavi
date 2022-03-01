from matplotlib import pyplot as plt
import librosa
from librosa import display
import numpy as np

def visualize_spec(self, spec, sr, dest, title=None): #TODO: fix the self parameters passaation 

    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        spec,
        # x_axis="time",
        sr=sr,
        cmap="coolwarm",
    )
    height = spec.shape[0]
    width = spec.shape[1]
    height_interval = int(height/10)
    width_interval = int(width/10)
    plt.yticks(np.arange(0, height, height_interval))
    plt.xticks(np.arange(0, width, width_interval))
    plt.colorbar()
    if title:
        plt.title(title)
        dest = dest + "__" + title
    plt.show()
    plt.savefig(dest + ".png")
    plt.close()

def visualize_spec_bis(spec, sr, dest, title=None): 

    fig = plt.figure(figsize=(20, 10))
    display.specshow(
        spec,
        # x_axis="time",
        sr=sr,
        cmap="coolwarm",
    )
    height = spec.shape[0]
    width = spec.shape[1]
    height_interval = int(height/10)
    width_interval = int(width/10)
    plt.yticks(np.arange(0, height, height_interval))
    plt.xticks(np.arange(0, width, width_interval))
    plt.colorbar()
    if title:
        plt.title(title)
        dest = dest + "__" + title
    plt.show()
    plt.savefig(dest + ".png")
    plt.close()


def visualize_audio(audio_c, dest, title=None, xlabel=None, ylabel=None):
    plt.figure()
    plt.plot(audio_c)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
        dest = dest + "__" + title
    plt.plot()
    plt.savefig(dest)
    plt.close()

def pad_sample(source, output_length): # TODO: same exists in preparer -> concile
    output_length = int(output_length)
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    frac = src_length / output_length
    if(frac < 0.5):
        # tile forward sounds to fill empty space
        cursor = 0
        while(cursor + src_length) < output_length:
            copy[cursor:(cursor + src_length)] = source[:]
            cursor += src_length
    else:
        copy[:src_length] = source[:]
    #
    return copy