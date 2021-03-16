
import subprocess

from .helpers import *


def test_wav(cycles, annotated_dict, root, sr, audio_length, overlap_threshold, dataset=None, tests_destination='tests/', test_files_destination ='tests/test_files/'): 
    # TESTING #1
    print("Running the first set of tests...")
    
    path = test_files_destination + "/"
    
    name = dataset + "_cycles.pkl"
    write_to_file(cycles, name, path + name)
    name = dataset + "_annotated_dict.pkl"
    write_to_file(annotated_dict, name, path)
    
    try:
        test_path = tests_destination + "/" + dataset + "_tests.py"
        proc = subprocess.run(["/opt/anaconda3/envs/ObjectDetection/bin/python", test_path, "{}".format(audio_length), "{}".format(overlap_threshold), 
                               "{}".format(sr), "{}".format(root)], check=True,)  # stdout=PIPE, stderr=PIPE,)
        print("The script passed all the tests.")
    except subprocess.CalledProcessError:
        print("The script failed to pass all the tests. ")
        sys.exit(1)

def test_spectrograms(spectrograms, height, width, dataset=None, tests_destination='tests/', test_files_destination='tests/test_files/'):
        # TESTING #2
        print("Running the second set of tests...")

        path = test_files_destination + "/"
        name = dataset + "_train_data.pkl"
        write_to_file(spectrograms[0], name, path + name)
        name = dataset + "_val_data.pkl"
        write_to_file(spectrograms[1], name, path + name)
        
        try:
            test_path = tests_destination + "/" + dataset + "_tests_2.py"
            proc = subprocess.run(["/opt/anaconda3/envs/ObjectDetection/bin/python", test_path,
                                "{}".format(height), "{}".format(width)], check=True)  # stdout=PIPE, stderr=PIPE,)
        except subprocess.CalledProcessError:
            print("The scripts failed to pass all the tests. ")
            sys.exit(1)