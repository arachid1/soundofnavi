from .SpecAugmenter import SpecAugmenter
from ..main import parameters
import numpy as np
import tensorflow as tf

class CutMixSpecAugmenter(SpecAugmenter):

    def __init__(self, samples, quantity, no_pad, label_one, label_two, minval, maxval):
        super().__init__(samples, quantity, no_pad, label_one, label_two, minval, maxval)
        self.name += "_cutmix"
    
    def augment_singular_sample(self, one, two, idx):
        
        image_one, label_one, filename_one = one
        image_two, label_two, filename_two = two
        
        # print(image_one)
        # print(label_one)
        # print(filename1)

        # print(image_two)
        # print(label_two)
        # print(filename2)

        alpha = [0.25]
        beta = [0.25]

        # lambda_value = sample_beta_distribution(1, alpha, beta)
        lambda_value = tf.random.uniform(shape=[], minval=self.minval, maxval=self.maxval)

        # lambda_value = lambda_value[0]

        boundaryx1, boundaryy1, target_h, target_w = self.get_box(lambda_value, parameters.shape)

        image_one = np.expand_dims(image_one, axis=-1)
        image_two = np.expand_dims(image_two, axis=-1)

        # Get a patch from the second image (`image_two`)
        crop2 = tf.image.crop_to_bounding_box(
            image_two, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image_two` patch (`crop2`) with the same offset
        image_two = tf.image.pad_to_bounding_box(
            crop2, boundaryy1, boundaryx1, parameters.shape[0], parameters.shape[1]
        )
        # Get a patch from the first image (`image_one`)
        crop1 = tf.image.crop_to_bounding_box(
            image_one, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image_one` patch (`crop1`) with the same offset
        img1 = tf.image.pad_to_bounding_box(
            crop1, boundaryy1, boundaryx1, parameters.shape[0], parameters.shape[1]
        )

        image_one = image_one - img1
        image = image_one + image_two

        # Adjust Lambda in accordance to the pixel ration
        lambda_value = 1 - (target_w * target_h) / (parameters.shape[0] * parameters.shape[1])
        lambda_value = tf.cast(lambda_value, tf.float32)

        label = lambda_value * label_one + (1 - lambda_value) * label_two
        label = label.numpy()
        
        image_one = np.squeeze(image_one)
        image_two = np.squeeze(image_two)
        image = np.squeeze(image)

        # print(image_one.shape)
        # self.visualize_spec(image_one, parameters.sr, "images_one")
        # self.visualize_spec(image_two, parameters.sr, "images_two")
        # self.visualize_spec(image, parameters.sr, "images")

        new_filename = self.generate_new_name(filename_one, filename_two, idx)
        # print(new_filename)

        return image, label, new_filename
        
    def get_box(self, lambda_value, shape):
        cut_rat = tf.math.sqrt(1.0 - lambda_value)

        cut_w = shape[1] * cut_rat  # rw
        cut_w = tf.cast(cut_w, tf.int32)

        cut_h = shape[0] * cut_rat  # rh
        cut_h = tf.cast(cut_h, tf.int32)

        cut_x = tf.random.uniform((1,), minval=0, maxval=shape[1], dtype=tf.int32)  # rx
        cut_y = tf.random.uniform((1,), minval=0, maxval=shape[0], dtype=tf.int32)  # ry

        boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, shape[1])
        boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, shape[0])

        bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, shape[1])
        bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, shape[0])

        target_h = bby2 - boundaryy1
        if target_h == 0:
            target_h += 1

        target_w = bbx2 - boundaryx1
        if target_w == 0:
            target_w += 1

        return boundaryx1, boundaryy1, target_h, target_w