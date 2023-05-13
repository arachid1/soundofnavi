from ..main import parameters as p
from collections import defaultdict

class Recording:
    def __init__(self, id, audio, label, patient):
        self.id = id
        self.audio = audio
        self.label = label
        self.logs_folder = ''
        self.patient = patient
        self.slices = defaultdict(lambda: None)
        
        # TODO: return fucnctions? for slice too?