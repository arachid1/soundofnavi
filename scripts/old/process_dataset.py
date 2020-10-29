import numpy as np
import os
import pyaudio
import wave
import shutil


def main():
    root = input(
        "Enter the path to the folder containing the ICBHI audio files: ")
    path_exists = os.path.isdir(root)
    if not (path_exists):
        print("The path given isn't a directory. ")
        exit()
    filenames = [s.split('.')[0]
                 for s in os.listdir(path=root) if '.wav' in s]
    print("Number of Files: {}".format(len(filenames)))
    for i, filename in enumerate(filenames):
        print("Name of file {}: {}".format(i + 1, filename))
        file_path = str(root + '/' + filename + '.wav')
        # open and play stream
        while True:
            f = wave.open(file_path, "rb")
            print("Length of file {}: {} seconds".format(
                i+1, f.getnframes()/f.getframerate()))
            print("Started reading file {}. ".format(i + 1))
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                            channels=f.getnchannels(),
                            rate=f.getframerate(),
                            output=True)
            data = f.readframes(f.getnframes())
            while data:
                stream.write(data)
                data = f.readframes(f.getnframes())
            stream.stop_stream()
            stream.close()
            print("Finished reading file {}. ".format(i + 1))
            relisten = input("Would you like to hear again? (y or n)")
            if relisten == 'y':
                continue
            elif relisten == 'n':
                break
            else:
                print("Invalid input. The audio is replaying. ")
                continue
        while True:
            delete = input("Would you like to delete the file ? (y or n)")
            if delete == 'y':
                logs = open("logs.txt", "a")
                logs.write(str(filename + '\n'))
                os.rename(file_path, str(
                    root + '/deleted/' + filename + '.wav'))
                os.rename(str(root + '/' + filename + '.txt'), str(
                    root + '/deleted/' + filename + '.txt'))
                cause = input(
                    "Why do you want to delete the file? (Enter 0 for no explanation)")
                if cause == 0:
                    cause = 'None'
                print("yo")
                logs.write(str(cause + '\n' + '\n'))
                logs.close()
                break
            elif delete == 'n':
                break
            else:
                print("Invalid input. Try again. ")
                continue

        p.terminate()

    # for filename in filenames:
    # while(True):

    # ask for path X
    # check that file exists X
    # for loop: X
    # print file name  X
    # print length X
    # print label
    # open file X
    # do you want to delete (or move) ? y vs no X
    # if yes, why?
    # if yes, write file name & why into a text file


if __name__ == "__main__":
    main()
