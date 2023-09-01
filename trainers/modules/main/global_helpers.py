from matplotlib import pyplot as plt
import librosa
from librosa import display
import numpy as np
from . import parameters as p
from PIL import Image
import os


def visualize_spec(
    self, spec, sr, dest, title=None
):  # TODO: fix the self parameters passaation
    fig = plt.figure(figsize=(20, 10))
    display.specshow(spec, sr=sr, cmap="coolwarm")
    height = spec.shape[0]
    width = spec.shape[1]
    height_interval = int(height / 10)
    width_interval = int(width / 10)
    plt.yticks(np.arange(0, height, height_interval))
    plt.xticks(np.arange(0, width, width_interval))
    plt.colorbar()
    if title:
        plt.title(title)
        dest = dest + "__" + title
    plt.show()
    plt.savefig(dest + ".png")
    plt.close()


# def visualize_spec_bis(spec, sr, dest, title=None):

#     fig = plt.figure(figsize=(20, 10))
#     display.specshow(
#         spec,
#         # x_axis="time",
#         sr=sr,
#         cmap="coolwarm",
#     )
#     height = spec.shape[0]
#     width = spec.shape[1]
#     height_interval = int(height/10)
#     width_interval = int(width/10)
#     plt.yticks(np.arange(0, height, height_interval))
#     plt.xticks(np.arange(0, width, width_interval))
#     plt.colorbar()
#     if title:
#         plt.title(title)
#         dest = dest + "__" + title
#     plt.show()
#     plt.savefig(dest + ".png")
#     plt.close()

# def visualize_audio(audio_c, dest, title=None, xlabel=None, ylabel=None):
#     plt.figure()
#     plt.plot(audio_c)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     if title:
#         plt.title(title)
#         dest = dest + "__" + title
#     plt.plot()
#     plt.savefig(dest)
#     plt.close()

# def pad_sample(source, output_length): # TODO: same exists in preparer -> concile
#     output_length = int(output_length)
#     copy = np.zeros(output_length, dtype=np.float32)
#     src_length = len(source)
#     frac = src_length / output_length
#     if(frac < 0.5):
#         # tile forward sounds to fill empty space
#         cursor = 0
#         while(cursor + src_length) < output_length:
#             copy[cursor:(cursor + src_length)] = source[:]
#             cursor += src_length
#     else:
#         copy[:src_length] = source[:]
#     #
#     return copy


def mel_spectrogram(audio, ax):
    """
    function to generate a mel spectrogram from audio
    """
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio,
            sr=p.sr,
            n_fft=p.n_fft,
            hop_length=p.hop_length,
            n_mels=p.n_mels,
            center=False,
        ),
        ref=np.max,
    )
    display = librosa.display.specshow(
        mel, x_axis="time", y_axis="mel", sr=p.sr, fmax=8000, ax=ax
    )
    # display = display.to_rgba(display.get_array().reshape((p.n_mels, -1, 1)))
    # print(display.shape)
    # print(type(display))
    # return display
    return None


def plot(data, dest, plot_title, subplot_data_func, subplot_title_func, nrows, ncols=3):
    """
    plots a list of data to a figure with a flexible number of elements, ultimately arranged in rows and columns,
    and saves the figure to a file
    """
    plt.figure(figsize=(15, 12))
    plt.suptitle(plot_title)
    plt.subplots_adjust(hspace=0.2)

    # calculate number of rows
    nrows = len(data) // ncols + (len(data) % ncols > 0)

    # loop through the length of tickers and keep track of index
    for n, el in enumerate(data):
        # add a new subplot iteratively using nrows and cols
        ax = plt.subplot(nrows, ncols, n + 1)

        obj = subplot_data_func(el, ax)
        if obj is not None:
            ax.plot(obj)

        # chart formatting
        ax.set_title(subplot_title_func(el))

    plt.savefig(dest)
    plt.close()
    return obj


def avg_and_std(audios, logs, name):
    print("yo")
    audios = np.array(audios)
    shape = list(audios.shape)
    shape[0] = 1
    shape = tuple(shape)
    print(shape)
    sum = np.zeros(shape=shape)
    for a in audios:
        print(a.shape)
        sum += a
    avg = sum / len(audios)
    std = np.reshape(np.std(audios, axis=0), newshape=shape)

    print(avg)
    print(avg.shape)
    print(std.shape)

    plt.xlim(-2, 1)
    plt.plot(list(range(0, len(avg))), avg, "x")
    plt.savefig(os.path.join(logs, name + "_avg.png"))

    plt.plot(list(range(0, len(std))), std, "x")
    plt.savefig(os.path.join(logs, name + "_std.png"))

    # avg = Image.fromarray(avg)
    # avg = avg.convert("RGB")
    # avg.save(os.path.join(logs, name + "_avg.png"))
    # std = Image.fromarray(std)
    # std = std.convert("RGB")
    # std.save(os.path.join(logs, name + "_std.png"))
    # pass
