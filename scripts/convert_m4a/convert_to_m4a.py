import os
import glob
from pydub import AudioSegment

def main():

    folder = 'perch_8000_10seconds'
    source_dir = '../../data/raw_audios/{}/'.format(folder)  # Path where the videos are located
    # source_dir = '../../data/raw_audios/test/'  # Path where the videos are located
    dest_dir = '../../data/m4a_raw_audios/{}/'.format(folder)

    source = os.path.join(source_dir, '*.wav')
    for video in glob.glob(source):
        m4a_filename = os.path.splitext(os.path.basename(video))[0] + '.m4a'
        dest = os.path.join(dest_dir, m4a_filename)
        audio = AudioSegment.from_file(video)
        # print(audio.duration_seconds == 10.0)
        audio.export(dest, format='ipod')

if __name__ == "__main__":
    main()