from ..main import parameters as p
from collections import defaultdict

class Slice:
    def __init__(self, id, audio, label, recording):
        self.id = id
        self.audio = audio
        self.label = label
        self.logs_folder = ''
        self.file_path = []
        self.recording = recording
        # self.spec = self.visuali(audio)