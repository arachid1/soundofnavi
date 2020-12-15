from functools import partial
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.python.lib.io import file_io
import pickle
import time
import concurrent
import multiprocessing
import argparse
import json

print("Tensorflow Version: {}".format(tf.__version__))


class class_accuracy(tf.keras.metrics.Metric):
    def __init__(self, name="val_accuracy", **kwargs):
        super(class_accuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name="acc", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.round(y_pred), tf.int32)
        matches = tf.equal(y_true, y_pred)
        matches = tf.map_fn(
            lambda x: tf.equal(tf.reduce_sum(
                tf.cast(x, tf.float32)), 2), matches
        )
        matches = tf.cast(matches, tf.float32)
        acc = tf.reduce_sum(matches) / \
            tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.accuracy.assign(acc)

    def result(self):
        return self.accuracy


def is_correct(label, output):
    # print(label)
    # print(output)
    output = output[0]
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    if label[0] == output[0] and label[1] == output[1]:
        return True
    return False


def write_to_file(data, path):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()

# predict_ops():


def predict(interpreter, input_details, output_details, sample):
    try:
        # api.predict_ops()
        start_time = time.time()
        input_data = sample[0]
        input_data = np.repeat(input_data[..., np.newaxis], 3, -1)
        input_data = np.reshape(input_data, (1,) + input_data.shape)
        # print(input_data.shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        label = [sample[1], sample[2]]
        print("Label: {}".format(label))
        print("Output: {}".format(output_data))
        correct_pred = is_correct(label, output_data)
        # if correct_pred == True:
        #     total_correct += 1
        # print(" Acc: %.0f " % ((total_correct/(i+1))*100))
        print("--- %.2f seconds ---" % (time.time() - start_time))
        print("------------")
        return 1
    except:
        print("error")


def main(train_file, saved_model_dir):

    file_stream = file_io.FileIO(train_file, mode="rb")
    data = pickle.load(file_stream)
    # convert = False
    # interpret = True
    tflite_dir = saved_model_dir + 'model.tflite'
    h5_dir = saved_model_dir + 'model.h5'
    pkl_dir = saved_model_dir + 'model.pkl'
    multiprocessed = False
    # if convert:
    #     all_data = [data[i][y][z] for i in range(len(data)) for y in range(
    #         len(data[i])) for z in range(len(data[i][y]))]
    #     np.random.shuffle(all_data)
    #     print("Length of entire dataset: {}".format(len(all_data)))
    #     class_acc = class_accuracy()
    #     print(h5_dir)
    #     model = load_model(h5_dir)
    #     print(model.summary())
    #     converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #     # print(converter)/Users/alirachidi/Documents/Sonavi_Labs/ObjectDetection/data/datasets/
    #     converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #     converter.representative_dataset = representative_dataset_gen(all_data)
    #     tflite_quant_model = converter.convert()
    #     print("Model converted.")

    #     print("TFLite mode: {}".format(tflite_dir))
    #     with open(tflite_dir, 'wb') as f:
    #         print("Writing TF Lite model to directory.")
    #         f.write(tflite_quant_model)
    #     write_to_file(tflite_quant_model, pkl_dir)
    # if interpret:
    validation_data = data[1][0] + data[1][1] + data[1][2] + data[1][3]
    np.random.shuffle(validation_data)
    print("Length of Val Set: {}".format(len(validation_data)))
    # print(tflite_dir)bash gc
    # print(tflite_dir)
    tflite_stream = file_io.FileIO(pkl_dir, mode="rb")
    tflite_data = pickle.load(tflite_stream)
    interpreter = tf.lite.Interpreter(model_content=tflite_data, num_threads=4)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    print("#### Input/Output details ####")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    # Adjusting the invoker for single inference
    input_shape = [1, 257, 251, 3]
    output_shape = [1, 2]
    interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
    interpreter.resize_tensor_input(
        output_details[0]['index'], output_shape)
    interpreter.allocate_tensors()
    print("#### New Input/Output details ####")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    # Calculcating accuracy over the entire validation s4t
    print("Interpreter is being invoked...")
    if multiprocessed:
        pool_size = 5
        p = multiprocessing.Pool(pool_size)
        func = partial(predict, interpreter, input_details, output_details)
        # print(len(validation_data))
        # print(type(validation_data[0][0]))
        # for i, sample in enumerate(validation_data):
        #     validation_data[i] = validation_data[i][0]
        # print(validation_data[0])
        # validation_data = [1, 35, 345, 2]
        p.map(func, validation_data)
        # for sample in items:
        #     pool.apply_async(worker, (item,))
        p.close()
        p.join()
    else:
        total_correct = 0
        for i in range(len(validation_data)):
            start_time = time.time()
            input_data = validation_data[i][0]
            input_data = np.repeat(input_data[..., np.newaxis], 3, -1)
            input_data = np.reshape(input_data, (1,) + input_data.shape)
            # print(input_data.shape)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            label = [validation_data[i][1], validation_data[i][2]]
            print("Label: {}".format(label))
            print("Output: {}".format(output_data))
            correct_pred = is_correct(label, output_data)
            if correct_pred == True:
                total_correct += 1
            print("Acc: %.1f percents" % ((total_correct/(i+1))*100))
            duration = time.time() - start_time
            print("--- %.2f seconds ---" % (duration))
            print("------------")
            # print(i % 10)
            # print((i % 10) == 0)
            if ((i % 10) == 0):
                time_left = ((len(validation_data) - i + 1) * duration)/60/60
                print("Excepted time left after {} elements: {:.2f} hours".format(
                    i + 1, time_left))
        print("Final Accuracy: {}".format(total_correct/len(validation_data)))
        print("Interpreter is no longer invoked.")

    # pool_size = 5
    # p = multiprocessing.Pool(pool_size)
    # func = partial(predict, interpreter, input_details, output_details)

    # # print(len(validation_data))
    # # print(type(validation_data[0][0]))
    # # for i, sample in enumerate(validation_data):
    # #     validation_data[i] = validation_data[i][0]
    # # print(validation_data[0])
    # validation_data = [1, 35, 345, 2]
    # p.map(func, validation_data)

    # # for sample in items:
    # #     pool.apply_async(worker, (item,))

    # p.close()
    # p.join()

    # # executor = concurrent.futures.ProcessPoolExecutor(10)
    # # futures = []
    # # for sample in validation_data:
    # #     future = executor.submit(predict, sample, interpreter)
    # #     print(future.result())
    # #     futures.append(future)
    # # concurrent.futures.wait(futures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    # print(arguments)
    job_dir = arguments.pop("job_dir")
    params = arguments.pop("params")
    params = json.loads(params)
    # print(params)
    train_file = arguments.pop("train_file")
    # print(train_file)
    saved_model_dir = params["SAVED_MODEL_DIR"]
    # tflite_name = params["TFLITE_NAME"]
    main(train_file, saved_model_dir)
