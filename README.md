# Classification_algorithm

classification_algorithm is a repository for classification of abnormal lung sounds by Sonavi Labs Inc.

# Installation

I will include all the necessary steps for installation, such as the google-cloud setup or pip libraries via requirements.txt, when necessary

# Where to find?

Data Generation

- Under scripts/dataset_generation, you will find config_run_dataset.sh, which configurates the data generation (i.e., sample rate, mel or log, augmentation) and calls generate_dataset.py with the necessary arguments
- generate_dataset.py will occasionally run tests found under scripts/dataset_generation/tests, which themselves call data objects (.pkl) under scripts/dataset_generation/tests/test_files, a folder not provided at the moment. You can avoid any issues by setting test to 0 in config_run_dataset.sh, meaning no tests will be run. 
- Paths for the data sources and destinations will need to be adjusted in config_run_dataset.sh. 

Best Models

- Under trainers/main/top_performing_models, you will find files named with the format id_filenumber_type_data_accuracy.py each containing their respective models

# Using and Analyzing Cochlear Preprocession Script (Python)

- Under cochlear_preprocessing, you will find wav2aud2.py along the filter orders and values (i.e., COCH_A, C0CH_B, p). 
- Before running any scripts, change the file_path variable (right under the main() function), which is the path to your wav file. 
- Running the file will generate and plot a spectrogram. Under the validation folder, the script will create a folder named after the wav file being processed. In the said directory, you will find the saved spectrogram, its numerical values in full_aud_spect.txt, and a folder for each column containing the arma filter, the half wave rectification, lateral inhibitory mask and temporal integration window values. 
- Libraries to be downloaded are included in cochlear_preprocessing/requirements.txt