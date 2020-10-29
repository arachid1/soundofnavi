# Classification_algorithm

classification_algorithm is a repository for classification of abnormal lung sounds by Sonavi Labs Inc.

# Installation

I will include all the necessary steps for installation, such as the google-cloud setup or pip libraries via requirements.txt, when necessary

# Where to find?

Data Generation

- Under scripts/dataset_generation, you will find config_run_dataset.sh, which configurates the data generation (i.e., sample rate, mel or log, augmentation) and calls generate_dataset.py with the necessary arguments. generate_dataset.py will occasionally run tests found under scripts/dataset_generation/tests, which themselves call data objects (.pkl) under scripts/dataset_generation/tests/test_files, a folder not provided at the moment. You can avoid any issues by setting test to 0 in config_run_dataset.sh, meaning no tests will be run. 
- Paths for the data sources and destinations will need to be adjusted in config_run_dataset.sh. 

Cochlear Preprocession Script (Python)

- Under cochlear_preprocessing, you will wav2aud2.py