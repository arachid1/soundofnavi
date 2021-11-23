import tensorflow as tf
from ..main import parameters


def spec_parser(spectrogram, label, shape, initial_channels, cuberooting, normalizing):

    # spectrogram = sample[0]
    # tf.print("inside parse")
    # label = sample[1]
    # tf.print(spectrogram)
    tf.print(tf.shape(spectrogram))
    # tf.print(label)
    spectrogram = tf.reshape(spectrogram, shape)
    # if cuberooting:
    #     spectrogram = tf.math.pow(spectrogram, 1/3)
    # if normalizing:
    #     max_value = tf.reduce_max(spectrogram)
    #     spectrogram = tf.math.divide(spectrogram, max_value)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.tile(spectrogram, [1, 1, initial_channels])
    # tf.print("spectrogram")
    # tf.print(spectrogram)
    return spectrogram, tf.cast(label, tf.float32)

def generate_timed_spec(spectrogram, label, shape, initial_channels, cuberooting, normalizing):

    spectrogram = tf.transpose(spectrogram, [1, 0])
    # spectrogram = tf.reshape(spectrogram, shape)
    # time_per_seq = int(shape[0]/parameters.n_sequences)
    # spectrogram = tf.slice(spectrogram, [0,0], [time_per_seq, shape[1]])
    spectrogram = tf.split(spectrogram, parameters.n_sequences)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.tile(spectrogram, [1, 1, 1, initial_channels])
    return spectrogram, tf.cast(label, tf.float32)

# def prepare_audio(audio, label, shape, initial_channels, cuberooting, normalizing):

#     spectrogram = tf.transpose(spectrogram, [1, 0])
#     # spectrogram = tf.reshape(spectrogram, shape)
#     # time_per_seq = int(shape[0]/parameters.n_sequences)
#     # spectrogram = tf.slice(spectrogram, [0,0], [time_per_seq, shape[1]])
#     spectrogram = tf.split(spectrogram, parameters.n_sequences)
#     spectrogram = tf.expand_dims(spectrogram, axis=-1)
#     spectrogram = tf.tile(spectrogram, [1, 1, 1, initial_channels])
#     return spectrogram, tf.cast(label, tf.float32)