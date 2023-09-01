
from ..main import parameters as p
from collections import defaultdict

class Patient:
    def __init__(self, id, dataset):
        self.id = id
        self.logs_folder = ''
        self.dataset = dataset
        self.recordings = defaultdict(lambda: None) 
        
    # TODO: get demographics metadata 
        
    ######### Return helper functions #########
    def return_slices_by_recording(self):
        d = defaultdict(lambda: [])
        for recording_id, recording in self.recordings.items():
            d[recording_id].append(recording.slices.values())
        return d
    
    def get_metadata(self):
        metadata =  self.dataset.metadata
        return metadata[metadata['Patient_id'] == int(self.id)]