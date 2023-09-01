from .parse_functions import *
from .helpers import visualize_spectrogram, load_text

import pandas as pd
import xlwt 
from xlwt import Workbook 
import glob
import os
import shutil
import numpy as np
import tensorflow as tf
import math
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, auc
import nlpaug.flow as naf
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
import random
from .helpers import *
# tf.config.experimental_run_functions_eagerly(True)

def normalize(spec):
  """Normalize input image channel-wise to zero mean and unit variance."""
  return spec / np.max(spec)

def augment_and_mix(path, job_version, wav_path, shape, augm_path, minval, maxval, severity=3, width=3, depth=-1, alpha=1.):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(
      np.random.dirichlet([alpha] * width))
#   m = np.float32(np.random.beta(alpha, alpha))
  m = np.random.uniform(low=minval, high=maxval)
#   print(m)
#   print("new")
#   print(path)
  image = load_text(path)
#   visualize_spectrogram(image, 8000, 'augmix_norm_mix/orig.png')

  mix = np.zeros(shape, dtype=np.float32)
  for i in range(width):
    d = depth if depth > 0 else np.random.randint(1, 4)
    for i in range(d):
        wav_folder = wav_path[i]
        new_path = path.replace(job_version, wav_folder) 
        if wav_folder == 'v42':
            new_path = new_path.replace('v44', 'v46')
        if wav_folder == 'v43':
            new_path = new_path.replace('v44', 'v45')
        if wav_folder == 'v48':
            new_path = new_path.replace('v44', 'v47')
        # print(new_path)
        image_aug = load_text(new_path)
    # Preprocessing commutes since all coefficients are convex
    mix += (ws[i] * image_aug)

  mix = normalize(mix)

#   visualize_spectrogram(mix, 8000, 'augmix_norm_mix/mix.png')
  mixed = (1 - m) * normalize(image) + m * mix
  file_name = path.split('/')[-1].split('.')[0]
#   print(file_name)
#   destination = os.path.join(augm_path, file_name)
  file_path = write_to_txt_file(mixed, augm_path, file_name)
#   visualize_spectrogram(mixed, 8000, 'augmix_norm_mix/mixed.png')
  return file_path

def apply_augmix_augmentation(train_samples, augmix_quantity, dataset_path, job_version, wav_path, label, shape, minval, maxval):

    augmented_indices = list(range(0, len(train_samples)))
    random.shuffle(augmented_indices)
    augmented_audios = [train_samples[i] for i in augmented_indices if (train_samples[i][1] == label)]
    random.shuffle(augmented_audios)
    augmented_audios = augmented_audios[:augmix_quantity]

    folder_id = 1
    subfolder = 'augmented_mix'
    augm_path = os.path.join(dataset_path, subfolder)
    while os.path.exists(augm_path):
        subfolder = 'augmented_mix_{}'.format(folder_id)
        augm_path = os.path.join(dataset_path, subfolder)
        folder_id += 1
    print("The path used for augmix augmentations is {}".format(augm_path))

    os.mkdir(augm_path)

    augmented_samples = []
    # counter = 0
    for sample, label in augmented_audios:
        # counter += 1
        # if counter == 1 or counter == 2 or counter == 3:
        #     continue
        sample = augment_and_mix(sample, job_version, wav_path, shape, augm_path, minval, maxval, depth=len(wav_path), alpha=0.2)
        # print("here")
        # print(label)
        augmented_samples.append([sample, label])
        # exit()

    return augmented_samples

def sample_beta_distribution(size, concentration_0, concentration_1):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

@tf.function
def mix_up(ds_one, ds_two, batch_size, minval, maxval, alpha=0.2):

    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    labels_one = tf.cast(labels_one, tf.float32)
    labels_two = tf.cast(labels_two, tf.float32)

    # l = sample_beta_distribution(batch_size, alpha, alpha)
    l = tf.random.uniform(shape=[], minval=minval, maxval=maxval)

    # x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))
    x_l = l
    x_l_2 = 1 - x_l

    images_one_mod = images_one * x_l
    images_two_mod = images_two * x_l_2

    images = images_one_mod + images_two_mod

    max_value = tf.reduce_max(images)
    images = tf.math.divide(images, max_value)

    labels = (labels_one * y_l) + (labels_two * (1 - y_l))

    # images = tf.squeeze(images, axis=-1)
    # tf.print(tf.shape(images))
    # labels = tf.squeeze(labels, axis=0)
    # tf.print(labels)

    return images, labels

def apply_mixup_augmentation(train_dataset, both_classes, repeat_factor, mixup_quantity, shape, minval, maxval, no_zeropad):

    if both_classes:
        if no_zeropad:
            train_dataset = train_dataset.filter(lambda x, y: tf.numpy_function(remove_zero_padded, [x, shape], tf.bool))
            print("length after removing zero-pads: " + str(len(list(train_dataset.as_numpy_iterator()))))
        pneumonia_train_dataset = train_dataset.filter(lambda x, y: y == 1)
        non_pneumonia_train_dataset = train_dataset.filter(lambda x, y: y == 0)
        # pneumonia_length = len(list(pneumonia_train_dataset.as_numpy_iterator()))
        # print(pneumonia_length)
        train_slice_one = pneumonia_train_dataset
        train_slice_one = train_slice_one.repeat(repeat_factor) 
        mixup_quantity = len(list(train_slice_one.as_numpy_iterator()))
        # print(mixup_quantity)
        train_slice_two = non_pneumonia_train_dataset.take(mixup_quantity)
        mixup_quantity = len(list(train_slice_two.as_numpy_iterator()))
        # print(mixup_quantity)
    else:
        train_slice_one = train_dataset.take(mixup_quantity) 
        train_slice_two = train_dataset.take(mixup_quantity)
    
    train_slice_one = train_slice_one.shuffle(mixup_quantity)
    train_slice_two = train_slice_two.shuffle(mixup_quantity)

    train_slice = tf.data.Dataset.zip((train_slice_one, train_slice_two))

    mixup_train_dataset = train_slice.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, 1, minval, maxval), num_parallel_calls=tf.data.AUTOTUNE)

    mixup_train_dataset = mixup_train_dataset.batch(1)
    # mixup_train_dataset = mixup_train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

    return mixup_train_dataset

# @tf.function
def get_box(lambda_value, shape):
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

# @tf.function
def cutmix(train_ds_one, train_ds_two, minval, maxval, shape):
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two

    alpha = [0.25]
    beta = [0.25]

    # lambda_value = sample_beta_distribution(1, alpha, beta)
    lambda_value = tf.random.uniform(shape=[], minval=minval, maxval=maxval)

    # lambda_value = lambda_value[0]

    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value, shape)
    if boundaryy1 >= 32:
        if boundaryy1 < 64:
            boundaryy1 = boundaryy1 - 32
        elif boundaryy1 < 96: 
            boundaryy1 = boundaryy1 - 64
        else:
            boundaryy1 = boundaryy1 - 96

    # Get a patch from the second image (`image2`)
    crop2 = tf.image.crop_to_bounding_box(
        image2, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image2` patch (`crop2`) with the same offset
    image2 = tf.image.pad_to_bounding_box(
        crop2, boundaryy1, boundaryx1, shape[0], shape[1]
    )
    # Get a patch from the first image (`image1`)
    crop1 = tf.image.crop_to_bounding_box(
        image1, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image1` patch (`crop1`) with the same offset
    img1 = tf.image.pad_to_bounding_box(
        crop1, boundaryy1, boundaryx1, shape[0], shape[1]
    )

    image1 = image1 - img1
    image = image1 + image2

    # Adjust Lambda in accordance to the pixel ration
    lambda_value = 1 - (target_w * target_h) / (shape[0] * shape[1])
    lambda_value = tf.cast(lambda_value, tf.float32)

    label = lambda_value * label1 + (1 - lambda_value) * label2
    # return image1, image2, image, label
    return image, label

def apply_cutmix_augmentation(train_dataset, both_classes, repeat_factor, cutmix_quantity, shape):

    if both_classes:
        pneumonia_train_dataset = train_dataset.filter(lambda x, y: y == 1)
        non_pneumonia_train_dataset = train_dataset.filter(lambda x, y: y == 0)
        train_slice_one = pneumonia_train_dataset.take(158)
        train_slice_one = train_slice_one.repeat(repeat_factor) 
        train_slice_two = non_pneumonia_train_dataset.take(cutmix_quantity)
    else:
        train_slice_one = train_dataset.take(cutmix_quantity) 
        train_slice_two = train_dataset.take(cutmix_quantity)
    
    train_slice_one = train_slice_one.shuffle(cutmix_quantity)
    train_slice_two = train_slice_two.shuffle(cutmix_quantity)

    train_slice = tf.data.Dataset.zip((train_slice_one, train_slice_two))

    cutmix_train_dataset = train_slice.map(lambda ds_one, ds_two: cutmix(ds_one, ds_two, shape), num_parallel_calls=tf.data.AUTOTUNE)

    return cutmix_train_dataset

def gen_chunks(lst, n, i):

    return lst[(i*n):(i+1)*n]

def apply_spec_augmentations(samples_to_augment, dataset_path, augmented_length, time_masking, frequency_masking, loudness, main_job_version, wav_path, factor=None, coverage_low_threshold=0.35, coverage_high_threshold=0.75, zone=(0.2, 0.8)):

    folder_id = 1
    subfolder = 'augmented'
    augm_path = os.path.join(dataset_path, subfolder)
    while os.path.exists(augm_path):
        subfolder = 'augmented_{}'.format(folder_id)
        augm_path = os.path.join(dataset_path, subfolder)
        folder_id += 1
    print("The path used for spec augmentations is {}".format(augm_path))

    os.mkdir(augm_path)
    portion_nbs = sum([time_masking, frequency_masking, loudness])
    portion_length = math.ceil(augmented_length/portion_nbs)


    fs = call_functions(time_masking, frequency_masking, loudness)

    augmented_samples = []

    counter = 0
    for i in range(portion_nbs):
        f = fs[i]
        chunks = gen_chunks(samples_to_augment, portion_length, i)
        random.shuffle(chunks)
        for c in chunks:
            recording, label = c
            spec = load_text(recording)
            # visualize_spectrogram(spec, 8000, "normal_{}".format(counter))
            if f.__name__ == "f_loudness":
                zone = (0.2, 0.8)
                coverage_low_threshold = 0.5
                coverage_high_threshold = 0.9
                coverage = random.uniform(coverage_low_threshold, coverage_high_threshold) 
                factor = (0.5, 2)
            elif f.__name__ == "f_frequency_masking":
                zone = (0, 1)
                coverage_low_threshold = 0.2
                coverage_high_threshold = 0.6
                coverage = random.uniform(coverage_low_threshold, coverage_high_threshold)
                factor = (15, 60)
            else:
                zone = (0.2, 0.8)
                coverage_low_threshold = 0.4
                coverage_high_threshold = 0.8
                coverage = random.uniform(coverage_low_threshold, coverage_high_threshold)
            spec = f(spec, coverage, factor, zone)
            # visualize_spectrogram(spec, 8000, "time_masking_test_{}".format(counter))
            source_folder = main_job_version
            for job_version in wav_path:
                if not (recording.find(job_version) == -1):
                    source_folder = job_version
            file_name = recording.split('/')[-1].split('.')[0]
            new_name = str(file_name + '___' + str(i) + '_' + source_folder + '_' + str(counter) + '.txt') # _augmentationfunc_sourcefolder_id
            file_path = os.path.join(augm_path, new_name)
            np.savetxt(file_path, spec, delimiter=',')
            augmented_samples.append((file_path, label))
            counter += 1
            
    return augmented_samples

def call_functions(time_masking, frequency_masking, loudness):
    fs = []
    if time_masking == 1:
        # print("time_masking")
        # print(time_masking)
        fs.append(f_time_masking)
    if frequency_masking == 1:
        # print("time_masking")
        # print(time_masking)
        fs.append(f_frequency_masking)
    if loudness == 1:
        fs.append(f_loudness)
    random.shuffle(fs)
    return fs

def f_time_masking(spec, coverage, factor, zone):
    
    aug = nas.TimeMaskingAug(zone=zone, coverage=coverage)
    aug_spec = aug.augment(spec)
    return aug_spec

def f_frequency_masking(spec, coverage, factor, zone=(0.5, 1)):
    
    aug = nas.FrequencyMaskingAug(zone=zone, factor=factor, coverage=coverage)
    aug_spec = aug.augment(spec)
    return aug_spec

def f_loudness(spec, coverage, factor, zone):
    
    aug = nas.LoudnessAug(zone=zone, factor=factor, coverage=coverage) 
    aug_spec = aug.augment(spec)
    return aug_spec

# def manage_augmentations(train_samples, wav_add, spec_add, wav_quantity, spec_quantity, job_version, wav_path, time_masking, frequency_masking, loudness, dataset_path, subfolder="augmented"):
#     if wav_add:
#         all_augmented_audios = []
#         for wav_folder in wav_path:
#             augmented_indices = random.sample(range(len(train_samples)), wav_quantity)
#             augmented_audios = [train_samples[i] for i in augmented_indices]
#             for i in range(len(augmented_audios)):
#                 augmented_audios[i][0] = augmented_audios[i][0].replace(job_version, wav_folder) # TODO: rename some variables
#             all_augmented_audios.extend(augmented_audios)
#         train_samples += all_augmented_audios
#         print("Length of training samples after audio augmentation: {}".format(len(train_samples)))
        
#     if spec_add:
#         augmented_indices = random.sample(range(len(train_samples)), spec_quantity)
#         augmented_specs = [train_samples[i] for i in augmented_indices] 
#         augmented_specs = apply_spec_augmentations(augmented_specs, os.path.join(dataset_path, subfolder), len(augmented_specs), time_masking, frequency_masking, loudness, job_version, wav_path)
#         train_samples += augmented_specs
#         print("Length of training samples after spec augmentation: {}".format(len(train_samples)))

#     train_labels = [label for _, label in train_samples] 
#     nb_train_pneumonia, nb_train_non_pneumonia = train_labels.count(1), train_labels.count(0)
#     print("-----------------------")
#     print("Number of train recordings: {} with pneumonia: {} and non-pneumonia: {}".format(len(train_samples), nb_train_pneumonia, nb_train_non_pneumonia))

#     return train_samples, train_labels


def manage_label_augmentations(train_samples, wav_add, spec_add, wav_quantity, spec_quantity, job_version, wav_path, time_masking, frequency_masking, loudness, dataset_path, label):
    if wav_add:
        all_augmented_audios = []
        for wav_folder in wav_path:
            augmented_indices = list(range(0, len(train_samples)))
            random.shuffle(augmented_indices)
            augmented_audios = [train_samples[i] for i in augmented_indices if (train_samples[i][1] == label)]
            random.shuffle(augmented_audios)
            augmented_audios = augmented_audios[:wav_quantity] # TODO: might have to add a clause for length later
            for i in range(len(augmented_audios)):
                # add other dataset
                augmented_audios[i][0] = augmented_audios[i][0].replace(job_version, wav_folder) # TODO: rename some variables
                if wav_folder == 'v42':
                    augmented_audios[i][0] = augmented_audios[i][0].replace('v44', 'v46')
                if wav_folder == 'v43':
                    augmented_audios[i][0] = augmented_audios[i][0].replace('v44', 'v45')
                if wav_folder == 'v48':
                    augmented_audios[i][0] = augmented_audios[i][0].replace('v44', 'v47')
            # print(augmented_audios[:45])
            # exit()
            all_augmented_audios.extend(augmented_audios)
        train_samples += all_augmented_audios
        print("Length of training samples after audio augmentation: {}".format(len(train_samples)))
    
    # print(spec_quantity)
    if spec_add:
        augmented_indices = list(range(0, len(train_samples)))
        random.shuffle(augmented_indices)
        augmented_specs = [train_samples[i] for i in augmented_indices if (train_samples[i][1] == label)]
        random.shuffle(augmented_specs)
        augmented_specs = augmented_specs[:spec_quantity]
        # print("len: " + str(len(augmented_specs)))
        # exit()
        augmented_specs = apply_spec_augmentations(augmented_specs, dataset_path, len(augmented_specs), time_masking, frequency_masking, loudness, job_version, wav_path)
        train_samples += augmented_specs
        print("Length of training samples after spec augmentation: {}".format(len(train_samples)))

    train_labels = [label for _, label in train_samples] 
    nb_train_pneumonia, nb_train_non_pneumonia = train_labels.count(1), train_labels.count(0)
    print("-----------------------")
    print("Number of train recordings: {} with pneumonia: {} and non-pneumonia: {}".format(len(train_samples), nb_train_pneumonia, nb_train_non_pneumonia))

    return train_samples, train_labels

def remove_zero_padded(x, shape):
    column = tf.gather(x, shape[1] - 1, axis=1)
    column_sum = tf.reduce_sum(column)
    # tf.print("here")
    # tf.print(column_sum)
    # tf.print(tf.math.count_nonzero(column))
    # x = x.numpy()
    # print(x)
    # print(x.shape)
    # if column_sum < 0.1:
    #     visualize_spectrogram(np.squeeze(x), 8000, 'zeropad_{}.png'.format(random.randint(0, 200)))
    return column_sum > 0.1