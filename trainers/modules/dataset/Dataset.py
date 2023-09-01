from ..main import parameters as p
from collections import defaultdict
from ..patient.Patient import Patient
from ..recording.Recording import Recording
from ..slice.Slice import Slice
import os
import librosa
import types
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
import shutil
from modules.main.global_helpers import *
from PIL import Image


# TODO: change structure? make it all dicts? i.e. lamdda of all objs?
class Dataset:
    """
    :param root: root folder of the dataset
    """

    def __init__(self, id, root, metadata_path, get_filenames):
        self.id = id
        self.root = root
        self.logs_root = (
            "/Users/alirachidi/Documents/Sonavi Labs/classification_algorithm/data_logs"
        )
        self.metadata_path = metadata_path
        self.metadata = self.build_metadata()
        self.get_filenames = types.MethodType(get_filenames, self)
        # self.logs_folder = ''
        self.patients = defaultdict(lambda: None)
        self.testing_files_registry = []

    ######### Workflow main functions #########

    def load_recordings(self):
        # filenames without .wav
        self.filenames = self.get_filenames()
        # if testing mode, calls hard-coded filenames for each dataset
        if p.testing:
            self.filenames = self.get_testing_filenames()
        # a dictionary of labels for ALL the filenames
        label_dict = self.build_label_dict()
        for filename in self.filenames:
            patient_id = self.get_patient_id(filename)
            audio = self.load_audio(filename)
            # accessing each of the dictonary for individual labels using patient id
            label = self.read_recording_label(label_dict, patient_id)
            # the filename must be stripped of its path (relative or absolute) and extension (.wav)
            # NOTE: except for BD?
            if patient_id in self.patients.keys():
                recording = Recording(filename, audio, label, self.patients[patient_id])
                self.patients[patient_id].recordings[filename] = recording
            else:
                patient = Patient(patient_id, self)
                recording = Recording(filename, audio, label, patient)
                patient.recordings[filename] = recording
                self.patients[patient_id] = patient

    def prepare_slices(self):
        for patient_id, patient in self.patients.items():
            for recording_id, recording in patient.recordings.items():
                self.slice_recording(recording)

    def slice_recording(self, recording):
        audio = recording.audio
        label = recording.label
        filename = recording.id
        rw_index = 0
        max_ind = len(audio)
        additional_step = True
        order = 0
        while additional_step:
            start = rw_index
            end = rw_index + p.audio_length
            audio_c, start_ind, end_ind = self.slice_array(
                start, end, audio, p.sr, max_ind
            )
            if not (
                abs(end_ind - start_ind) == (p.audio_length * p.sr)
            ):  # ending the loop
                additional_step = False
                if abs(start_ind - end_ind) < (
                    p.sr * p.audio_length * Decimal(0.5)
                ):  # disregard if LOE than fraction of the audio length (<=1 for 0.5 using 2sec, <=5 for 10sec)
                    continue
                else:  # 0 pad if more than half of audio length
                    audio_c = self.pad_sample(audio_c, p.sr * p.audio_length)
            # find_label returns an integer (-1 if none)
            dataset_id = recording.patient.dataset.id
            if p.mode == "cw" and (
                dataset_id == "Icbhi"
                or dataset_id == "Perch"
                or dataset_id == "Antwerp"
            ):  # FIXME
                audio_c_label = self.read_slice_label(label, start_ind, end_ind)
            else:
                audio_c_label = self.read_slice_label(label)
            audio_c_filename = self.generate_slice_filename(
                filename, order, additional_step
            )
            slice = Slice(audio_c_filename, audio_c, audio_c_label, recording)
            recording.slices[audio_c_filename] = slice
            rw_index = start + p.step_size
            order += 1

    ######### Loading workflow helpers #########
    def load_audio(self, filename):
        path = os.path.join(self.root, filename + ".wav")
        data, ___ = librosa.load(path, sr=p.sr)
        return data

    def get_filenames(self):  # overwritten
        pass

    def get_testing_filenames(self):  # overwritten
        pass
        # TODO: write a function that reads a files of custom testing files from testing_file_registry

    def build_label_dict(self):  # overwritten
        pass

    def read_recording_label(self):  # overwritten
        pass

    def get_patient_id(self):  # overwritten
        pass

    ######### Preparing workflow helpers #########
    def generate_slice_filename(self, filename, order, additional_step):
        """
        generates filename for chunk by adding '_i' with i = 0-indexed order of the chunk.
        If last chunk,  i == 99
        """
        if not additional_step:
            order = 99
        return filename + "_" + str(order)

    def slice_array(self, start, end, raw_data, sample_rate, max_ind=10e5):
        """
        slices the [start, end] chunk of raw_data
        """
        start_ind = min(int(start * sample_rate), max_ind)
        end_ind = min(int(end * sample_rate), max_ind)
        return raw_data[start_ind:end_ind], start_ind, end_ind

    def pad_sample(self, source, output_length):
        """
        pad source to output_length
        """
        output_length = int(output_length)
        copy = np.zeros(output_length, dtype=np.float32)
        src_length = len(source)
        frac = src_length / output_length
        if frac < 0.5:
            # tile forward sounds to fill empty space
            cursor = 0
            while (cursor + src_length) < output_length:
                copy[cursor : (cursor + src_length)] = source[:]
                cursor += src_length
        else:
            copy[:src_length] = source[:]
        return copy

    def read_slice_label(self, recording_dict):  # overwritten
        pass

    ######### Return helper functions #########
    def return_recordings_by_patient(self):
        d = {}
        for patient_id, patient in self.patients.items():
            d[patient_id] = patient.recordings.values()
            # d[p.id] = p.recordings
        return d

    def return_slices_by_patient(self):
        d = defaultdict(lambda: [])
        for patient_id, patient in self.patients.items():
            for recording_id, recording in patient.recordings.items():
                d[patient_id].append(recording.slices.values())
        return d

    def return_slices_by_recording_by_patient(self):
        d = defaultdict(lambda: {})
        for patient_id, patient in self.patients.items():
            d[patient_id] = patient.return_slices_by_recording()
        return d
        # uses return_slices_by_measurement in Patient(), which uses return_slices() in Measurement()

    ######### Analysis helper functions #########

    def get_dataset_profile():
        pass

    def build_metadata(self):  # overwritten
        pass

    def analyze(self):
        # resetting the logs folder
        # TODO: pass option for it?
        dataset_logs_root = os.path.join(self.logs_root, self.id, p.mode)
        if os.path.exists(dataset_logs_root):
            shutil.rmtree(dataset_logs_root)
        os.makedirs(dataset_logs_root)

        # TODO: generalize to non-mel specs
        # TODO: add build_metadata() and get_dataset_profile()

        for patient in self.patients.values():
            patient_logs = os.path.join(dataset_logs_root, str(patient.id))
            os.mkdir(patient_logs)
            patient_slices_audios = []
            patient_recordings_audios = []
            for recording in patient.recordings.values():
                recording_logs = os.path.join(patient_logs, str(recording.id))
                os.mkdir(recording_logs)
                # full recording: audio
                recording_audio = plot(
                    [recording],
                    dest=str(recording_logs + "/{}_audio".format(str(recording.id))),
                    plot_title=str(recording.id),
                    subplot_data_func=lambda l, ax: l.audio,
                    subplot_title_func=lambda l: l.label,
                    nrows=1,
                    ncols=1,
                )
                # full recording: mel
                plot(
                    [recording],
                    dest=str(recording_logs + "/{}_audio".format(str(recording.id))),
                    plot_title=str(recording.id),
                    subplot_data_func=lambda l, ax: mel_spectrogram(l.audio, ax),
                    subplot_title_func=lambda l: l.label,
                    nrows=1,
                    ncols=1,
                )

                patient_recordings_audios.append(recording_audio)
                recording_slices_audios = []
                for slice in recording.slices.values():
                    slice_logs = os.path.join(recording_logs, str(slice.id))
                    os.mkdir(slice_logs)
                    # slice: audio
                    slice_audio = plot(
                        [slice],
                        dest=str(slice_logs + "/{}_audio".format(str(slice.id))),
                        plot_title=str(slice.id),
                        subplot_data_func=lambda l, ax: l.audio,
                        subplot_title_func=lambda l: l.label,
                        nrows=1,
                        ncols=1,
                    )
                    # slice: mel
                    plot(
                        [slice],
                        dest=str(slice_logs + "/{}_mel".format(str(slice.id))),
                        plot_title=str(slice.id),
                        subplot_data_func=lambda l, ax: mel_spectrogram(l.audio, ax),
                        subplot_title_func=lambda l: l.label,
                        nrows=1,
                        ncols=1,
                    )
                    recording_slices_audios.append(slice_audio)

                # TODO slices of recordings: avg and std of audio
                avg_and_std(
                    recording_slices_audios,
                    recording_logs,
                    str(recording.id + "_slices"),
                )
                patient_slices_audios.extend(recording_slices_audios)

                # TODO slices of recordings: avg and std of spec

            # TODO slices of patient: avg and std of audio
            avg_and_std(
                patient_slices_audios, patient_logs, str(patient.id + "_slices")
            )

            # TODO slices of patient: avg and std of spec

            # TODO recordings of patient: avg and std of audio

            # TODO recordings of patient: avg and std of spec

        # dataset avgs
