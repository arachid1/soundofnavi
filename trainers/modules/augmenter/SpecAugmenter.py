from .Augmenter import Augmenter
from ..main import parameters
import random

class SpecAugmenter(Augmenter):

    def __init__(self, samples, quantity, no_pad, label_one, label_two, minval, maxval):
        super().__init__(samples, quantity, no_pad)
        self.label_one = label_one
        self.label_two = label_two
        self.minval = minval
        self.maxval = maxval
        self.name = "spec"

    def augment_all_samples(self):
        samples_to_augment = self.get_samples_to_augment()
        idx = 0
        for one, two in samples_to_augment:
            aug_sample = self.augment_singular_sample(one, two, idx)
            self.aug_samples.append(aug_sample)
            # if idx == 1:
            #     break
            idx += 1

    def get_samples_to_augment(self):
        
        label_one_samples = self.return_samples_with_label(self.label_one) # TODO: if same label, make sure it's different elements
        label_two_samples = self.return_samples_with_label(self.label_two)
        # print("in get samples")
        # print(len(label_one_samples))
        # print(len(label_two_samples))
        desired_length = min(len(label_one_samples), len(label_two_samples))
        label_one_samples = self.match_length(label_one_samples, desired_length)
        label_two_samples = self.match_length(label_two_samples, desired_length)
        # print(len(label_one_samples))
        # print(len(label_two_samples))
        self.shuffle(label_one_samples)
        self.shuffle(label_two_samples)
        samples_to_augment = zip(label_one_samples, label_two_samples)
        return samples_to_augment
    
    def generate_new_name(self, filename_one, filename_two, idx):
        return filename_one + "__" + filename_two + "__" + self.name + "__" + str(idx)