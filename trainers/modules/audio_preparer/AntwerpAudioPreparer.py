from .IcbhiAudioPreparer import IcbhiAudioPreparer, find_cw_label
from ..main import parameters
import types
from decimal import *

class AntwerpAudioPreparer(IcbhiAudioPreparer):
    def __init__(self, samples, name, mode):
        super().__init__(samples, name, mode)
        if self.mode == "cw":
            self.find_label = types.MethodType(find_cw_label, self) 

    def return_patient_id(self, filename):

        for el in filename.split('_'):
            try:
                patient_id = int(el)
            except ValueError:
                continue
            break
        return patient_id
        # return None