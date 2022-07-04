from ..main import parameters
# from ..main.global_helpers import visualize_spec
from ..main.global_helpers import visualize_spec_bis
import tensorflow as tf
# from .helpers import *
import numpy as np


# np.set_printoptions(precision=3)

class visualizationCallback(tf.keras.callbacks.LambdaCallback):

    def __init__(self, samples, save_frontend=True):
        self.samples = samples
        self.folder = "{}/others".format(parameters.job_dir)
        self.save_frontend = save_frontend
        self.initial_specs = []
        self.final_specs = []
        self.first_weights = None
        self.last_weights = None

    def on_train_end(self, logs=None):
        print(len(self.final_specs))
        print(len(self.initial_specs))
        for i in range(len(self.initial_specs)):
            try:
                diff = self.final_specs[i][0] - self.initial_specs[i][0]
            except IndexError:
                print(IndexError)
            visualize_spec_bis(diff, sr=parameters.sr, dest="{}/diff_{}".format(self.folder, i), title="label_{}_name_{}".format(self.initial_specs[i][1], self.initial_specs[i][2]))
        
        
        if parameters.distillation:
            current_weights = self.model.student._frontend.weights
        else:
            current_weights = self.model._frontend.weights

        print("Initial weights")
        print(self.first_weights)
        print("Post training weights")
        print(current_weights)
        print("Weight difference")
        # print(np.array(self.first_weights).shape)
        # print(np.array(self.model._frontend.weights).shape)
        try:
            print(np.array(current_weights) - np.array(self.first_weights))
        except ValueError:
            print(ValueError)

    def on_epoch_begin(self, epoch, logs=None):

        try:
            if epoch == 20 or epoch == 10 or epoch == 30:
                print("Mel loss: {}".format(self.model.mel_loss))
                print("sinc loss: {}".format(self.model.sinc_loss))
        except AttributeError:
            pass

        self.final_specs = []
        for i, sample in enumerate(self.samples):
            audio = sample[0]
            audio = np.expand_dims(audio, axis=0)
            try:
                if parameters.distillation:
                    spec = self.model.student(audio, False, True)
                else:
                    spec = self.model(audio, False, True)
            except ValueError as e:
                print ('error type: ', type (e))
                print(audio)
                print(type(audio))
                continue
            spec = np.swapaxes(np.squeeze(spec.numpy()), 0, 1)
            if epoch == 0:
                self.initial_specs.append((spec, sample[1], sample[2]))
            self.final_specs.append((spec, sample[1], sample[2]))
                # pair.append((spec, sample[1], sample[2]))
            visualize_spec_bis(spec, sr=parameters.sr, dest="{}/spec_{}_epoch_{}".format(self.folder, i, epoch), title="label_{}_name_{}".format(sample[1], sample[2]))
            
        if epoch == 0:
            if parameters.distillation:
                self.first_weights = self.model.student._frontend.weights
            else:
                self.first_weights = self.model._frontend.weights
            
        print("here")
        if parameters.distillation:
            self.model.student.save(parameters.job_dir, epoch)
        else:
            self.model.save(parameters.job_dir, epoch)
        print("here2")
        #     self.model._frontend.save_weights(parameters.job_dir + "/teacher_frontend_{}.h5".format(epoch))
        # self.model._functional.save_weights(parameters.job_dir + "/teacher_functional_{}.h5".format(epoch))

