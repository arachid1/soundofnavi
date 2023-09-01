from .Augmenter import Augmenter
from ..main import parameters
import random

class AudioAugmenter(Augmenter):

    def __init__(self, samples, quantity, label, no_pad, minval, maxval):
        super().__init__(samples, quantity, no_pad)
        self.label = label
        self.minval = minval
        self.maxval = maxval
        self.name = "audio"
    
    def augment_all_samples(self):
        samples_to_augment = self.get_samples_to_augment()
        idx = 0
        for sample_to_augment in samples_to_augment:
            aug_sample = self.augment_singular_sample(sample_to_augment, idx)
            self.aug_samples.append(aug_sample)
            idx += 1

    def get_samples_to_augment(self):
        augmented_indices = list(range(0, len(self.samples)))
        random.shuffle(augmented_indices)
        if self.label == -1:
            samples_to_augment = [self.samples[i] for i in augmented_indices]
        else:
            samples_to_augment = [self.samples[i] for i in augmented_indices if (self.samples[i][1] == self.label)]
        random.shuffle(samples_to_augment)
        samples_to_augment = self.match_length(samples_to_augment, len(samples_to_augment))
        return samples_to_augment
    
    def generate_new_name(self, filename, idx):
        return filename + "__" + self.name + "__" + str(idx)