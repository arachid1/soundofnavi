from .AudioPreparer import AudioPrepaper
from .IcbhiAudioPreparer import find_cw_label #, find_times, find_labels
import types
from decimal import *

class PerchAudioPreparer(AudioPrepaper):
    def __init__(self, samples, name, mode):
        super().__init__(samples, name, mode)
        if self.mode == "cw":
            self.find_label = types.MethodType(find_cw_label, self) 

    # def find_label(self, recording_dict):
    #     print(recording_dict)
    #     exit()
    #     if recording_dict['Diagnosis'] == 'Pneumonia':
    #         return 1
    #     else:
    #         return 0
    
    def return_patient_id(self, filename):
        return filename.split('-')[0]
    
    # def return_all_samples_by_patient(self):
    #     pass
        
    #     # print(self.chunked_samples[0])
    #     samples_by_patients = {}
    #     # patient_ids = []
    #     for s in self.chunked_samples:
    #         print(s)
    #         patient_id = int(s[0][2].split('_')[0])
    #         if patient_id not in samples_by_patients:
    #             samples_by_patients[patient_id] = s
    #         else:
    #             samples_by_patients[patient_id] += s
    #     samples_by_patients = list(samples_by_patients.values())
    #     self.chunked_samples = samples_by_patients
    #     return self.chunked_samples 
    #     # print(self.samples)
    #     # samples_by_patients = []
    #     # for i in range(1, 113):
    #     #     patient_samples = [s in self.samples if s[2].startswith("BP{}".format(i)) or s[2].startswith("DP{}".format(i)) or s[2].startswith("EP{}".format(i))]
    #     #     assert (len(patient_samples) == 3)
    #     #     samples_by_patients.append(patient_samples)
    #     # self.samples = samples_by_patients
    #     # return self.samples

# def find_cw_label(self, recording_dict, start, end):
#     times, ___ = find_times(recording_dict, parameters.sr, True)
#     # print(times)
#     return find_labels(times, start, end, parameters.overlap_threshold, parameters.audio_length, parameters.sr)

# def find_times(recording_annotations, sample_rate, first):
#     times = []
#     for i in range(len(recording_annotations.index)):
#         row = recording_annotations.loc[i]
#         c_start = Decimal('{}'.format(row['Start']))
#         c_end = Decimal('{}'.format(row['End']))
#         if (c_end - c_start < 1):
#             continue
#         if (first):
#             rw_index = c_start
#             first = False
#         times.append((c_start * sample_rate, c_end * sample_rate,
#                       row['Crackles'], row['Wheezes']))  # get all the times w labels
#     return times, rw_index

# def find_labels(times, start, end, overlap_threshold, audio_length, sample_rate):
#     overlaps = []
#     for time in times:
#         if start > time[0]:  # the window starts after
#             if start >= time[1]:  # no overlap
#                 continue
#             if end <= time[1]:  # perfect overlap: window inside breathing pattern
#                 # shouldn't happen since window is 2 sec long
#                 if (time[1] - time[0]) < (overlap_threshold * audio_length * sample_rate):
#                     continue
#                 else:
#                     return time[2], time[3]
#             else:  # time[1] < end: partial overlap, look at next window c)
#                 if (time[1] - start) < (overlap_threshold * audio_length * sample_rate):
#                     continue
#                 else:
#                     overlaps.append(time)
#         else:  # the window starts before
#             if end <= time[0]:  # no overlap
#                 continue
#             if end < time[1]:  # partial overlap b)
#                 if (end - time[0]) < (overlap_threshold * audio_length * sample_rate):
#                     continue
#                 else:
#                     overlaps.append(time)
#             else:  # end > time[1]: perfect overlap: breathing pattern inside of window
#                 # shouldn't happen since all windows are 1 sec or more
#                 if (time[1] - time[0]) < (overlap_threshold * audio_length * sample_rate):
#                     continue
#                 else:
#                     overlaps.append(time)
#     return list(process_overlaps(overlaps))

# def process_overlaps(overlaps):
#     crackles = sum([overlap[2] for overlap in overlaps])
#     wheezes = sum([overlap[3] for overlap in overlaps])
#     crackles = 1.0 if (not(crackles == 0)) else 0.0
#     wheezes = 1.0 if (not(wheezes == 0)) else 0.0
#     return crackles, wheezes
