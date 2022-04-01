# Dataset Table 

|   | Pneumonia| Crackles and Wheezes |
| ------------- | ------------- | ------------- |
| ICBHI  | X |  X |
| PERCH  |  |  X |
| JORDAN  | X | X |
| ANTWERP  |  | X |
| ANTWERP SIM  |  | X |
| BD  | X |  |


NOTE: As of 11/23, only Pneumonia problem for Icbhi/Jordan has been implemented. To be extended by adding training on BD for pneumonia + other datasets in CW.

NOTE: as of January 2022, you can train models inside the crackles/wheezes trainer (e.g., trainers/cw) with all of its datasets. 

# The Pneumonia Problem
## Set-up

Step 1) Retrieve data and place it inside 'classification_algorithm/data'

Firstly, 
```
gcloud auth login
```
When logged in, create empty 'data' folder. Then, run:
```
gcloud compute scp --recurse classification:/home/alirachidi/classification_algorithm/data data --zone us-central1-a
```

2) Install python 3.7.9 and libraries (see requirements.txt)
3) Edit local_gc_exec.sh according to your needs (see instructions inside file)
4) RUN: bash local_gc_exec.sh

What are some important components we will be using? 
- most important library used: trainers/modules
- train files for pneumonia training: trainers/main/models/train$.py, following a train$ format, with $ an integer (refer to train1.py as the template)
- utilities for train file: trainers/modules/main, which contains the following that we will learn more about: 

    - a) helpers.py: most of the functions called inside train$.py, such as load_audios, will be called from there. It helps keep the main training file tidy. 

    - b) parameters.py: a module (which is imported im most folders) to keep track of MOST 40+ (!!) parameters across files (for example, to allow accessing parameters.sr  or parameters.n_fft inside file A or B) and reflect modifications everywhere in a synchronized manner.

## Training

### important training files! 
main/train0 is the template is where you want to start from.

main/train1000 is used to reproduce our results from the **best model** from July/August 2020. It has an extra 'validation_only' parameters which, if set to true, print the validation results by loading the model. You can also find all the details from when it was trained inside best_model_cache(2360)/ at the root of the repo, including report.txt, tns.txt, its log 2360.out file, the saved models, some spectrograms...

### inside the files...

In local_gc_exe.sh, you have now indicated the appropriate trainer for the mode (i.e., trainers/main or trainers/cw) you want to work on, a training file and other important elements. Let's take a look inside your training file.

The following is an important section that you shouldn't have to modify for the most part. Its main objective with parameters.init() is to create a parent folder for caching. Be aware that it deletes any prior cache folder that corresponds to that file, so for example, running local_gc_exec.sh for train2356 will delete any exisiting cache/pneumonia/train2356 folder and initialize a new one. In other words, one file = one cache folder = one or more jobs. It also seeds our libraries, defines training mode, etc. 

```

if __name__ == "__main__":
    
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    seed_everything() # seeding np, tf, etc
    arguments = parameters.parse_arguments()
    parameters.init()
    parameters.mode = "pneumonia" if arguments["mode"] == "main" else arguments["mode"]
    parameters.n_classes = 2 if parameters.mode == "cw" else 1
    parameters.file_dir = os.path.join(parameters.cache_root, parameters.mode, os.path.basename(__file__).split('.')[0])
    parameters.description = arguments["description"]
    testing_mode(int(arguments["testing"])) # if true, adds _testing to the cache folder, sets a small number of epochs, smaller dataset, turn off a few settings in the callback, etc
    # works to set up the parent folder (i.e., overwrites if already exists) + works well with initialize_job above
    initialize_file_folder()
    print("-----------------------")
    .
    .
    .
```

In the following section, launch_job() is called. 

As you can see in the function header below, it first runs initialize_job(), which creates a child folder for the job (i.e., first job with id 1 goes goes into folder 1 in the parent cache folder, like cache/pneumonia/train2356/1) and handles other important tasks, like creating subfolders (i.e.,  "tp" or "tn") or incrementing parameters, such as job_id that is super important for caching as we just saw. 

Then, it runs train_model(), which is the last, and MAIN, function inside any file. It takes the parameters as described in the comments. The comments and the section on how the modules library work should help you understand its magic. 

````
    .
    .
    .
    ###### set up used for spec input models (default)
    parameters.hop_length = 254
    parameters.shape = (128, 311)
    parameters.n_sequences = 9
    spec_aug_params = [
        ["mixup", {"quantity" : 0.2, "no_pad" : False, "label_one" : 0, "label_two" : 1, "minval" : 0.3, "maxval" : 0.7}],
    ]
    audio_aug_params = [["augmix", {"quantity" : 0.2, "label": -1, "no_pad" : False, "minval" : 0.3, "maxval" : 0.7, "aug_functions": [shift_pitch, stretch_time]}]]

    launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, mixednet, spec_aug_params, audio_aug_params, spec_parser)
    
    # to run another job, add a line to modify whatever parameters, and rerun a launch_job function as many times as you want!
    # parameters.hop_length = 512
    # launch_job({"Bd": 0, "Jordan": 1, "Icbhi": 1, "Perch": 0, "Ant": 0, "SimAnt": 0,}, model9, spec_aug_params, audio_aug_params, spec_parser)
````

```
def launch_job(datasets, model, spec_aug_params, audio_aug_params, parse_function):
    # parameters: 
    # dictionary:  datasets to use, 
    # model:  imported from modules/models.py, 
    # augmentation parameters:  (No augmentation if empty list is passed), 
    # parse function:  (inside modules/parsers) used to create tf.dataset object in create_tf_dataset
    initialize_job()
    train_model(datasets, model, spec_aug_params, audio_aug_params, parse_function)
```

Finally, we must talk about the role of some of the components necessary for training. 

A) The first is modules, which will be covered in a section below. 

B) The second is parameters.py, which resides inside modules/main, and is a super important file. Think of it as an object accessible everywhere to extract virtually ANY parameter from a long list in a synchronized way. For example, it allows me to access parameters.sr inside train$.py BUT also inside my callback. Another instance where it's critical is, every time we run launch_job (and therefore initialize_job), it updates the job_id parameter to reflect the current job being executed (so for example, switching from the first job with id 1 to the second job with id 2), and the appropriate paths, etc. 

C) the third is models. You can write your model inside the train file train$.py, but I'd recommend writing inside and importing your model from modules/models/model$.py, with $ the number of the model. each model$.py has access to inside core.py, which has the custom mixednet layers, inverted residual layers, etc, so you HAVE to write inside models.py to use those elements. You can use mixednet.py as the template to copy/paste.

D) More to come on augmentation.

LASTE NOTE ON GPU/CPU: CUDA_DEVICES_VISIBLE is set to -1 or 0/1 inside local_gc_exec.sh to run in, respectively, CPU and GPU mode. GPUs vs non-GPUs runs may require slight modifications but that should all be debuggable in train$.py


## The Crackles/Wheezes problem

It's essentially very similar to pneumonia. However, I still have to implement augmentation and class weights for crackles/wheezes. the template for cw is cw/train0, equivalent to main/train0.

TODOs
## Add Bangladesh for pneumonia
## Augmentation and class weights for crackles wheezes
## Details on Augmentation
## Details on our audio library, modules
