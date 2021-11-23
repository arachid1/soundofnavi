import unittest
from decimal import *
import numpy as np
import pickle
from tensorflow.python.lib.io import file_io
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
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


class generate_dataset_tests(unittest.TestCase):

    cycle_list_path = 'tests_files/icbhi_cycles.pkl'
    cycle_list_stream = file_io.FileIO(cycle_list_path, mode="rb")
    cycle_list = pickle.load(cycle_list_stream)
    rec_annotations_dict_path = 'tests_files/icbhi_annotated_dict.pkl'
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

    def test_labelling_1(self):
        # falls outside
        times = ((4000, 4500, 1, 0), (4500, 5000, 0, 0), (5000, 5200, 0, 1))
        start = 3800
        end = 3900
        output = find_labels(times, start, end, self.test_threshold,
                             self.test_window_length, self.test_sample_rate)
        self.assertEqual(output[0], 0)
        self.assertEqual(output[1], 0)

    def test_labelling_2(self):
        # falls outside
        times = ((4000, 4500, 1, 0), (4500, 5000, 0, 0), (5000, 5200, 0, 1))
        start = 5300
        end = 7200
        output = find_labels(times, start, end, self.test_threshold,
                             self.test_window_length, self.test_sample_rate)
        self.assertEqual(output[0], 0)
        self.assertEqual(output[1], 0)

    def test_labelling_3(self):

        # testing thresholds (overlap: enough)
        times = ((4000, 4500, 1, 0), (4500, 5000, 0, 0), (5000, 10200, 0, 1))
        start = 3400
        end = 12300
        output = find_labels(times, start, end, self.test_threshold,
                             self.test_window_length, self.test_sample_rate)
        self.assertEqual(output[0], 0)
        self.assertEqual(output[1], 1)

    def test_labelling_4(self):
        # testing thresholds (overlap: enough)
        times = ((3400, 4500, 1, 0), (4500, 5000, 0, 0), (5000, 10200, 0, 1))
        start = 3400
        end = 12300
        output = find_labels(times, start, end, self.test_threshold,
                             self.test_window_length, self.test_sample_rate)
        self.assertEqual(output[0], 0)
        self.assertEqual(output[1], 1)

    def test_labelling_5(self):
        # testing thresholds (overlap: not enough)
        times = ((4000, 4500, 1, 0), (4500, 5000, 0, 0), (5000, 10200, 0, 1))
        start = 4300
        end = 12300
        output = find_labels(times, start, end, self.test_threshold,
                             self.test_window_length, self.test_sample_rate)
        self.assertEqual(output[0], 0)
        self.assertEqual(output[1], 1)

    def test_labelling_6(self):
        # testing thresholds (overlap: not enough)
        times = ((4000, 4500, 1, 0), (4500, 5000, 0, 0), (5000, 5300, 0, 1))
        start = 4300
        end = 12300
        output = find_labels(times, start, end, self.test_threshold,
                             self.test_window_length, self.test_sample_rate)
        self.assertEqual(output[0], 0)
        self.assertEqual(output[1], 0)

    def test_labelling_7(self):
        # testing thresholds (overlap: enough)
        times = ((4000, 4500, 0, 0), (4500, 7000, 1, 0), (7000, 9300, 0, 0))
        start = 4300
        end = 12300
        output = find_labels(times, start, end, self.test_threshold,
                             self.test_window_length, self.test_sample_rate)
        self.assertEqual(output[0], 1)
        self.assertEqual(output[1], 0)

    def test_get_sliced_samples_1(self):
        file_name = '148_1b1_Al_sc_Meditron'
        slices = get_icbhi_samples(
            self.rec_annotations_dict[file_name], file_name, self.root, self.test_sample_rate, 0, audio_length=2, step_size=1)
        self.assertEqual(len(slices), 18 + 2)

    def test_find_times_1(self):
        file_name = '148_1b1_Al_sc_Meditron'
        times = find_times(
            self.rec_annotations_dict[file_name], self.test_sample_rate, True)
        self.assertEqual(test_continuity(times[0]), True)
        self.assertEqual(times[1], Decimal('0.364'))
        self.assertEqual(test_time_labels(times[0]), [
                         (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)])

    def test_get_sliced_samples_2(self):
        file_name = '107_2b4_Al_mc_AKGC417L'
        slices = get_icbhi_samples(
            self.rec_annotations_dict[file_name], file_name, self.root, self.test_sample_rate, 0, audio_length=2, step_size=1)
        self.assertEqual(len(slices), 18 + 2)
        self.assertEqual(test_time_labels(slices[2:], True), [(1, 0), (1, 0), (1, 0), (1, 0), (1, 1), (1, 1), (
            1, 1), (1, 1), (1, 0), (1, 1), (0, 1), (1, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)])

    def test_find_times_2(self):
        file_name = '107_2b4_Al_mc_AKGC417L'
        times = find_times(
            self.rec_annotations_dict[file_name], self.test_sample_rate, True)
        self.assertEqual(test_continuity(times[0]), True)
        self.assertEqual(times[1], Decimal('1.018'))
        self.assertEqual(test_time_labels(times[0]), [
                         (1, 0), (1, 0), (1, 1), (1, 0), (0, 1), (1, 0), (1, 0), (1, 0)])

    def test_get_sliced_samples_3(self):
        file_name = '112_1p1_Pl_sc_Litt3200'
        slices = get_icbhi_samples(
            self.rec_annotations_dict[file_name], file_name, self.root, self.test_sample_rate, 0, audio_length=2, step_size=1)
        self.assertEqual(len(slices), 27 + 2)
        self.assertEqual(test_time_labels(slices[2:], True), [(1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (
            1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)])

    def test_find_times_3(self):
        file_name = '112_1p1_Pl_sc_Litt3200'
        times = find_times(
            self.rec_annotations_dict[file_name], self.test_sample_rate, True)
        self.assertEqual(test_continuity(times[0]), True)
        self.assertEqual(times[1], Decimal('2.3806'))

    def test_padding(self):
        source = np.random.rand(7503)
        self.assertEqual(generate_padded_samples(source, self.test_sample_rate *
                                                 self.test_window_length).shape, (self.test_sample_rate * self.test_window_length,))
        source = np.random.rand(2340)
        self.assertEqual(generate_padded_samples(source, self.test_sample_rate *
                                                 self.test_window_length).shape, (self.test_sample_rate * self.test_window_length,))
        return None

    def test_equality_1(self):
        # print("test_equality_1")
        file_name = '104_1b1_Al_sc_Litt3200'  # starts at 0
        slices = get_icbhi_samples(
            self.rec_annotations_dict[file_name], file_name, self.root, self.test_sample_rate, 0, audio_length=2, step_size=1)
        (rate, data) = read_wav_file(os.path.join(
            self.root, file_name + '.wav'), self.test_sample_rate)
        start_ind = 0
        for i in range(2, len(slices) - 1):  # don't check the last one bc usually padded
            chunk = slices[i][0]
            # print(chunk)
            correct_chunk = data[start_ind *
                                 self.test_sample_rate: (start_ind + 2) * self.test_sample_rate]
            # correct_chunk = np.true_divide(correct_chunk, np.abs(np.max(data)))
            # print(correct_chunk)
            comparison = correct_chunk == chunk
            equal_arrays = comparison.all()
            self.assertTrue(equal_arrays)
            start_ind += 1

    def test_equality_4(self):
        # print("test_equality_4")
        file_name = '109_1b1_Al_sc_Litt3200'  # starts at 0
        slices = get_icbhi_samples(
            self.rec_annotations_dict[file_name], file_name, self.root, self.test_sample_rate, 0, audio_length=2, step_size=1)
        (rate, data) = read_wav_file(os.path.join(
            self.root, file_name + '.wav'), self.test_sample_rate)
        start_ind = 0
        for i in range(2, len(slices) - 1):  # don't check the last one bc usually padded
            chunk = slices[i][0]
            correct_chunk = data[start_ind *
                                 self.test_sample_rate: (start_ind + 2) * self.test_sample_rate]
            # correct_chunk = np.true_divide(correct_chunk, np.abs(np.max(data)))
            comparison = correct_chunk == chunk
            equal_arrays = comparison.all()
            self.assertTrue(equal_arrays)
            start_ind += 1

    def test_equality_2(self):
        # print("test_equality_2")
        file_name = '104_1b1_Pl_sc_Litt3200'  # starts at 0
        slices = get_icbhi_samples(
            self.rec_annotations_dict[file_name], file_name, self.root, self.test_sample_rate, 0, audio_length=2, step_size=1)
        (rate, data) = read_wav_file(os.path.join(
            self.root, file_name + '.wav'), self.test_sample_rate)
        start_ind = 0
        for i in range(2, len(slices) - 1):  # don't check the last one bc usually padded
            chunk = slices[i][0]
            correct_chunk = data[start_ind *
                                 self.test_sample_rate: (start_ind + 2) * self.test_sample_rate]
            # correct_chunk = np.true_divide(correct_chunk, np.abs(np.max(data)))
            comparison = correct_chunk == chunk
            equal_arrays = comparison.all()
            self.assertTrue(equal_arrays)
            start_ind += 1

    def test_equality_3(self):
        # print("test_equality_3")
        file_name = '109_1b1_Al_sc_Litt3200'  # starts at 0
        slices = get_icbhi_samples(
            self.rec_annotations_dict[file_name], file_name, self.root, self.test_sample_rate, 0, audio_length=2, step_size=1)
        (rate, data) = read_wav_file(os.path.join(
            self.root, file_name + '.wav'), self.test_sample_rate)
        start_ind = 0
        for i in range(2, len(slices) - 1):  # don't check the last one bc usually padded
            chunk = slices[i][0]
            correct_chunk = data[start_ind *
                                 self.test_sample_rate: (start_ind + 2) * self.test_sample_rate]
            # correct_chunk = np.true_divide(correct_chunk, np.abs(np.max(data)))
            comparison = correct_chunk == chunk
            equal_arrays = comparison.all()
            self.assertTrue(equal_arrays)
            start_ind += 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_dataset_tests.root = sys.argv.pop()
        # print(generate_dataset_tests.test_window_length)
        generate_dataset_tests.test_sample_rate = int(sys.argv.pop())
        generate_dataset_tests.test_threshold = Decimal(sys.argv.pop())
        generate_dataset_tests.test_window_length = Decimal(sys.argv.pop())
    unittest.main()
