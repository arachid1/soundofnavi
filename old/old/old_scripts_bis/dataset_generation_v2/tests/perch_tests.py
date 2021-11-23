
import unittest
from decimal import *
import sys
import numpy as np
import pickle
from tensorflow.python.lib.io import file_io
sys.path.append(
    '/Users/alirachidi/Documents/Sonavi_Labs/ObjectDetection/trainers/main')
from generate_dataset import *



def test_continuity(times):
    for i in range(1, len(times)):
        start = times[i][0]
        end = times[i-1][1]
        if start != end:
            return False
    return True


def test_time_labels(times, slices=False):
    labels = []
    for time in times:
        if slices:
            labels.append((time[1], time[2]))
        else:
            labels.append((time[2], time[3]))
    return labels


class perch_tests(unittest.TestCase):

    cycle_list_path = 'tests_files/perch_cycle_list.pkl'
    cycle_list_stream = file_io.FileIO(cycle_list_path, mode="rb")
    cycle_list = pickle.load(cycle_list_stream)
    rec_annotations_dict_path = 'tests_files/perch_rec_annotations_dict.pkl'
    rec_annotations_dict_stream = file_io.FileIO(
        rec_annotations_dict_path, mode="rb")
    rec_annotations_dict = pickle.load(rec_annotations_dict_stream)
    test_window_length = None
    test_threshold = None
    test_sample_rate = None
    root = None

    def test_length(self):
        for cycle in self.cycle_list:
            self.assertEqual(
                Decimal('{}'.format(len(cycle[0]))), self.test_window_length * self.test_sample_rate)

    def test_padding(self):
        source = np.random.rand(7800)
        self.assertEqual(generate_padded_samples(source, self.test_sample_rate *
                                                 self.test_window_length).shape, (self.test_sample_rate * self.test_window_length,))
        source = np.random.rand(2340)
        self.assertEqual(generate_padded_samples(source, self.test_sample_rate *
                                                 self.test_window_length).shape, (self.test_sample_rate * self.test_window_length,))
        return None

    def test_get_samples_1(self):
        file_name = 'B00653-03_F_2.94000062_1'
        chunk = get_perch_sample(
            self.rec_annotations_dict[file_name], file_name, self.root, self.test_sample_rate, 0, self.test_window_length)
        self.assertEqual(len(chunk), 2 + 1)
        self.assertEqual(len(chunk[2][0]),
                         self.test_window_length * self.test_sample_rate)
        self.assertEqual((chunk[2][1], chunk[2][2]), (0, 0))

    def test_get_samples_2(self):
        file_name = 'B00666-03_F_8.28117845_1'
        chunk = get_perch_sample(
            self.rec_annotations_dict[file_name], file_name, self.root, self.test_sample_rate, 0, self.test_window_length)
        self.assertEqual(len(chunk), 2 + 1)
        self.assertEqual(len(chunk[2][0]),
                         self.test_window_length * self.test_sample_rate)
        self.assertEqual((chunk[2][1], chunk[2][2]), (0, 1))

    def test_get_samples_3(self):
        file_name = 'G01182-03_F_7.09558964_1'
        chunk = get_perch_sample(
            self.rec_annotations_dict[file_name], file_name, self.root, self.test_sample_rate, 0, self.test_window_length)
        self.assertEqual(len(chunk), 2 + 1)
        self.assertEqual(len(chunk[2][0]),
                         self.test_window_length * self.test_sample_rate)
        self.assertEqual((chunk[2][1], chunk[2][2]), (1, 0))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        perch_tests.root = sys.argv.pop()
        perch_tests.test_sample_rate = int(sys.argv.pop())
        perch_tests.test_threshold = Decimal(sys.argv.pop())
        perch_tests.test_window_length = Decimal(
            sys.argv.pop())
    else:
        print("Sorry, no argument received.")
        exit()
    unittest.main()
