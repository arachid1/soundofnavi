from .AudioAugmenter import AudioAugmenter
# from ..parse_functions.parse_functions import generate_spec
from ..main import parameters
from ..main.global_helpers import pad_sample
# import types
import numpy as np
import tensorflow as tf

def skip_init(cls): # TODO: move somewhere better
    actual_init = cls.__init__
    cls.__init__ = lambda *args, **kwargs: None
    instance = cls()
    cls.__init__ = actual_init
    return instance

class AugMixAudioAugmenter(AudioAugmenter):

    def __init__(self, samples, quantity, label, no_pad, minval, maxval, aug_functions, severity=3, width=3, depth=-1, alpha=1.):
        super().__init__(samples, quantity, label, no_pad, minval, maxval)
        self.aug_functions = aug_functions
        self.severity = severity
        self.width = width
        self.depth = len(self.aug_functions)
        self.alpha = alpha
        self.spec_generator = skip_init(SpecGenerator)
        self.name += "_augmix"
        
    def augment_singular_sample(self, sample, idx):
        """
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
        audio = sample[0]
        label = sample[1]
        filename = sample[2]
        new_filename = self.generate_new_name(filename, idx)

        ws = np.float32(
            np.random.dirichlet([self.alpha] * self.width))
        m = np.random.uniform(low=self.minval, high=self.maxval)
    
        image = self.spec_generator.generate_mel_spec(audio)
        # self.visualize_spec(image, parameters.sr, "new_augmix/orig")

        mix = np.zeros(parameters.shape, dtype=np.float32)
        for i in range(self.width):
            image_aug = audio.copy()
            d = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for y in range(d):
                # image_aug = pad_sample(image_aug, int(parameters.sr * parameters.audio_length))
                f = self.aug_functions[y]
                image_aug = f(image_aug)
            # Preprocessing commutes since all coefficients are convex
            image_aug = self.spec_generator.generate_mel_spec(image_aug)
            # image_aug = self.normalize(image_aug)
            # self.visualize_spec(image_aug, parameters.sr, "new_augmix/{}_{}".format(i, y))
            mix += (ws[i] * image_aug)

        #   visualize_spectrogram(mix, 8000, 'augmix/mix.png')
        # image = normalize(image)
        mixed = (1 - m) * image + m * mix
        # self.visualize_spec(mixed, parameters.sr, "new_augmix/mixed")
        # print(new_filename)
        return mixed, label, new_filename

    def normalize(self, el):
        return el / np.max(el)