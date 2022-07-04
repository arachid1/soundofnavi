from .SpecAugmenter import SpecAugmenter
from ..main import parameters
import numpy as np

class MixUpSpecAugmenter(SpecAugmenter):

    def __init__(self, samples, quantity, no_pad, label_one, label_two, minval, maxval):
        super().__init__(samples, quantity, no_pad, label_one, label_two, minval, maxval)
        self.name += "_mixup"
    
    def augment_singular_sample(self, one, two, idx):

        images_one, labels_one, filename_one = one
        # print(images_one)
        # print(labels_one)
        # print(filename_one)
        images_two, labels_two, filename_two = two
        # print(images_two)
        # print(labels_two)
        # print(filename_two)
        # exit()

        # labels_one = tf.cast(labels_one, tf.float32)
        # labels_two = tf.cast(labels_two, tf.float32)

        # l = tf.random.uniform(shape=[], minval=0.3, maxval=0.7)
        l = np.random.uniform(low=self.minval, high=self.maxval)
        # print(l)

        # x_l = tf.reshape(l, (batch_size, 1, 1, 1))
        # y_l = tf.reshape(l, (batch_size, 1))
        # x_l_2 = 1 - x_l

        images_one_mod = images_one * l
        images_two_mod = images_two * (1 - l)
        
        images = images_one_mod + images_two_mod

        # images = images / np.max(images)

        # max_value = tf.reduce_max(images)
        # images = tf.math.divide(images, max_value)

        labels = (labels_one * l) + (labels_two * (1 - l))
        # self.visualize_spec(images_one, parameters.sr, "images_one")
        # self.visualize_spec(images_two, parameters.sr, "images_two")
        # self.visualize_spec(images, parameters.sr, "images")

        new_filename = self.generate_new_name(filename_one, filename_two, idx)
        # print(new_filename)

        return images, labels, new_filename
        