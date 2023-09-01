from ..main import parameters
from ..main.global_helpers import visualize_spec
import types
import random

class Augmenter:
    def __init__(self, samples, quantity, no_pad):
        self.samples = samples
        self.quantity = quantity
        self.no_pad = no_pad
        self.visualize_spec = types.MethodType(visualize_spec, self) 
        self.name = None
        self.aug_samples = []
        self.shuffle(self.samples)
    
    def augment_all_samples(self):
        pass

    def augment_singular_sample(self):
        pass
    
    def visualize_spec(self):
        pass
    
    def normalize(self, el):
        pass
    
    def generate_new_name(self):
        pass
    
    def return_samples_with_label(self, label):
        samples_with_label = []
        random.shuffle(self.samples)
        for i, sample in enumerate(self.samples):
            if self.satisfies_condition(sample, label):
                samples_with_label.append(self.samples[i])
        return samples_with_label
    
    def satisfies_condition(self, sample, label):
        condition = (sample[1] == label) 
        if self.no_pad:
            condition = condition and (not (sample[2].endswith("99"))) 
        return condition
    
    def match_length(self, label_samples, smallest_length):
        random.shuffle(label_samples)
        initial_length = len(label_samples)
        desired_length = int(smallest_length * self.quantity)
        if desired_length > initial_length:
            i = 0
            while len(label_samples) < desired_length:
                index = i % initial_length 
                label_samples.append(label_samples[i])
                i += 1
        else:
            label_samples = label_samples[:desired_length]
        return label_samples
    
    def shuffle(self, to_shuffle):
        random.shuffle(to_shuffle)
    
    def return_all_samples(self):
        return self.aug_samples