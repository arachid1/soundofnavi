
import unittest
from decimal import *
import sys
import numpy as np
import pickle
from tensorflow.python.lib.io import file_io
sys.path.append(
    '/Users/alirachidi/Documents/Sonavi_Labs/ObjectDetection/trainers/main')
from generate_dataset import *


# def test_continuity(times):
#     for i in range(1, len(times)):
#         start = times[i][0]
#         end = times[i-1][1]
#         if start != end:
#             return False
#     return True


# def test_time_labels(times, slices=False):
#     labels = []
#     for time in times:
#         if slices:
#             labels.append((time[1], time[2]))
#         else:
#             labels.append((time[2], time[3]))
#     return labels


class generate_dataset_tests(unittest.TestCase):

    # cycle_list_path = 'temp/cycle_list.pkl'
    # cycle_list_stream = file_io.FileIO(cycle_list_path, mode="rb")
    # cycle_list = pickle.load(cycle_list_stream)
    # rec_annotations_dict_path = 'temp/rec_annotations_dict.pkl'
    # rec_annotations_dict_stream = file_io.FileIO(
    #     rec_annotations_dict_path, mode="rb")
    # rec_annotations_dict = pickle.load(rec_annotations_dict_stream)
    # test_window_length = None
    # test_threshold = None
    # test_sample_rate = None
    # root = None
    train_data_path = 'tests_files/perch_train_data.pkl'
    train_data_stream = file_io.FileIO(train_data_path, mode="rb")
    train_data = pickle.load(train_data_stream)
    val_data_path = 'tests_files/perch_val_data.pkl'
    val_data_stream = file_io.FileIO(val_data_path, mode="rb")
    val_data = pickle.load(val_data_stream)
    height = None
    width = None

    def test_shapes(self):
        for train in self.train_data:
            for element in train:
                self.assertEqual(element[0].shape, (self.height, self.width))

        for test in self.val_data:
            for element in test:
                self.assertEqual(element[0].shape, (self.height, self.width))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_dataset_tests.width = int(sys.argv.pop())
        generate_dataset_tests.height = int(sys.argv.pop())
    else:
        print("Sorry, no argument received.")
        exit()
    unittest.main()
