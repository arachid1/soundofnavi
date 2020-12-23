from .modules.helpers import *
from .modules.generators import *
from .modules.metrics import *
from .modules.callbacks import *
from .core import *

import argparse
import pickle
import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K

from tensorflow.python.lib.io import file_io
import json

print("Tensorflow Version: {}".format(tf.__version__))

def conv2d(N_CLASSES, SR, BATCH_SIZE, LR, SHAPE, WEIGHT_DECAY, LL2_REG, EPSILON):

    KERNEL_SIZE = (3, 3)
    POOL_SIZE = (2, 2)
    PADDING = "same"
    CHANNELS = 32
    DROPOUT = 0.1
    DENSE_LAYER = 32
    i = layers.Input(shape=SHAPE, batch_size=BATCH_SIZE)
    x = layers.BatchNormalization()(i)
    tower_1 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(32, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(128, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(128, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(128, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(256, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(256, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(256, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(512, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(512, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(512, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(512, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(512, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(1024, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(1024, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(1024, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(1024, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(1024, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    # delete above

    model = Model(inputs=i, outputs=o, name="conv2d")
    opt = tf.keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=EPSILON, decay=WEIGHT_DECAY, amsgrad=False
    )
    class_acc = class_accuracy()
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            'accuracy'
        ],
    )
    return model

def train_model(train_file, **args):
    logs_path = job_dir + "/logs/"
    print("-----------------------")
    print("Using module located at {}".format(__file__[-30:]))
    print("Using train_file located at {}".format(train_file))
    print("Using logs_path located at {}".format(logs_path))
    print("-----------------------")

    # setting variables
    print("Collecting Variables...")

    n_classes = params["N_CLASSES"]

    n_epochs = params["N_EPOCHS"]

    sr = params["SR"]
    lr = params["LR"]
    batch_size = params["BATCH_SIZE"]

    ll2_reg = params["LL2_REG"]
    weight_decay = params["WEIGHT_DECAY"]

    epsilon = params["EPSILON"]

    shape = tuple(params["SHAPE"])

    es_patience = params["ES_PATIENCE"]
    min_delta = params["MIN_DELTA"]

    initial_channels = params["INITIAL_CHANNELS"]
    shape = shape + (initial_channels, )
    
    save = bool(params["SAVE"])
    add_tuned = bool(params["ADD_TUNED"])

    model_params = {
        "N_CLASSES": n_classes,
        "SR": sr,
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "SHAPE": shape,
        "WEIGHT_DECAY": weight_decay,
        "LL2_REG": ll2_reg,
        "EPSILON": epsilon
    }

    factor = params["FACTOR"]
    patience = params["PATIENCE"]
    min_lr = params["MIN_LR"]

    lr_params = {
        "factor": factor,
        "patience": patience,
        "min_lr": min_lr
    }

    print("Model Parameters: {}".format(model_params))
    print("Learning Rate Parameters: {}".format(lr_params))
    print("Early Stopping Patience: {}".format(params["ES_PATIENCE"]))
    print("-----------------------")

    # data retrieval
    print("Collecting data...")
    file_stream = file_io.FileIO(train_file, mode="rb")
    data = pickle.load(file_stream)

    _train = [
        data[0][i] for i in range(0, len(data[0]))
    ]
    
    _test = [
        data[1][i] for i in range(0, len(data[1]))
    ]
    
    info(_train, _test)
    
    # data preparation
    train_data = [
        sample
        for label in _train
        for sample in label
    ]
    validation_data = [
        sample for label in _test for sample in label
    ]
    
    callback_params = {
        "validation_data": validation_data, 
        "shape": shape, 
        "n_classes": n_classes, 
        "sr": sr, 
        "save": save, 
        "add_tuned": add_tuned, 
        "es_patience": es_patience, 
        "min_delta": min_delta,
        "job_dir": job_dir
    }
    
    generator_params = {
        "train_data": train_data, 
        "sr": sr, 
        "n_classes": n_classes, 
        "shape": shape, 
        "batch_size": batch_size, 
        "initial_channels": initial_channels, 
    }

    # generators
    tg = data_generator(**generator_params)

    # callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="accuracy", **lr_params
    )
    confusion_m_callback = cm_callback(**callback_params)

    # model setting
    model = conv2d(**model_params)

    model.summary()

    weights = None

    if bool(params["CLASS_WEIGHTS"]):
        print("Initializing weights...")
        y_train = label_data(validation_data)
        weights = class_weight.compute_class_weight(
            "balanced", np.unique(y_train), y_train)

        weights = {i: weights[i] for i in range(0, len(weights))}
        print("weights = {}".format(weights))
        
    model.fit(
            tg,
            epochs=n_epochs,
            verbose=2,
            callbacks=[tensorboard_callback,
                       confusion_m_callback,
                       reduce_lr_callback,
                       #    early_stopping_callback
                       ],
            class_weight=weights,
    )

    model.save("model.h5")

    # Save model.h5 on to google storage
    with file_io.FileIO("model.h5", mode="rb") as input_f:
        with file_io.FileIO(job_dir + "/model.h5", mode="w+") as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    seed_everything()
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        "--train-file",
        help="GCS or local paths to training data",
        required=True
    )

    parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints and export models",
        required=True,
    )
    parser.add_argument(
        "--params",
        help="parameters used in the model and training",
        required=True,
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop("job_dir")
    params = arguments.pop("params")
    params = json.loads(params)
    train_model(**arguments)
