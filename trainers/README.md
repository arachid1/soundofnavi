# Training

1) Upload data and place 'data' folder inside classification_algorithm
gcloud auth login
initializr 'data' folder
gcloud compute scp --recurse classification:/home/alirachidi/classification_algorithm/data data --zone us-central1-a
2) Install python 3.7.9 and libraries (see requirements.txt)
3) Edit local_gc_exec.sh according to your needs (see instructions inside file)
4) RUN: bash local_gc_exec.sh

access to important components (from inside trainers):
- most important library used: main/models/conv/modules
- train files for pneumonia training: main/models/__.py
- utilities for train file: main/models/conv/modules/main, which contains
a) helpers.py: most of the functions called inside __.py, like load_audios, will be called from there
b) parameters.py: a module (which is imported im most folders) to keep track of ALL parameters across files (for example, to allow accessing parameters.sr inside file A or B) and reflect modifications everywhere accordingly

In local_gc_exe.sh, you have indicated the appropriate trainer for the mode (or folder/task) you want to work on, a training file and other important elements. Let's take a look inside your training file. 

This an important section that you shouldn't have to modify for the most part. Its main objective is to creates a parent folder for file cache. Be aware that it deletes any prior cache folder that corresponds to that file, so for example, running local_gc_exec.sh for train2356 will delete any exisiting cache/train2356 folder and initialize a new one. It also seeds our libraries, defines training mode, etc. 

```
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    seed_everything() # seeding np, tf, etc
    arguments = parameters.parse_arguments()
    parameters.init()
    parameters.mode = "pneumonia"
    parameters.file_dir = os.path.join(parameters.cache_root, parameters.mode, os.path.basename(__file__).split('.')[0])
    parameters.description = arguments["description"]
    testing_mode(int(arguments["testing"]))
    # works to set up the folder (i.e., overwrites for testing folders, duplication for non-testing folders) 
    # and works well with initialize_job, which initializes each job inside the file (i.e, creates all the subfolders like tp/tn/gradcam/etc, file saving conventions, etc)
    initialize_file_folder()
    print("-----------------------")

```

Here, the function to look at is launch_job. 
It first runs initialize_job(), which creates a child folder for the job (i.e., first job with id 1 goes goes into folder 1 in the parent cache folder, like train2356/cache/1) and handles other important tasks, like creating subfolders (i.e.,  "tp" or "tn") or incrementing parameters, like the job id that is super important for caching as we just saw. 
Then, it runs train_model(), which is the last, and MAIN, function inside any file. It takes the parameters as described in the comments. The comments and the section on the backbone of the library used should help you understand its magic. 


````
    ###### set up used for spec input models (default)
    parameters.hop_length = 254
    parameters.shape = (128, 311)
    parameters.n_sequences = 9
    spec_aug_params = [
        ["mixup", {"quantity" : 0.2, "no_pad" : False, "label_one" : 0, "label_two" : 1, "minval" : 0.3, "maxval" : 0.7}],
    ]
    audio_aug_params = [["augmix", {"quantity" : 0.2, "label": -1, "no_pad" : False, "minval" : 0.3, "maxval" : 0.7, "aug_functions": [shift_pitch, stretch_time]}]]
    # parameters: 
    # dictionary:  datasets to use, 
    # model:  imported from modules/models.py, 
    # augmentation parameters:  (No augmentation if empty list is passed), 
    # parse function:  (inside modules/parsers) used to create tf.dataset object in create_tf_dataset
    launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 0, "Perch": 0, "Ant": 0, "SimAnt": 0,}, mixednet, spec_aug_params, audio_aug_params, spec_parser)

    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!
````

Finally, we must talk about the last few components necessary for training. The first is models/conv/modules, which

The first is parameters.py, which resides inside modules/main, and is a super important file that allows us to have one copy of every parameter needed across all files and 

parameters.py and helpers.py inside modules.
pneumonia only
models/core/gpu

