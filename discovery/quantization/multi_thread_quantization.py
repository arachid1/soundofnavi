from functools import partial
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.python.lib.io import file_io
import pickle
import time
import concurrent
import multiprocessing
multiprocessing


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


def representative_dataset_gen(all_data):
    num_calibration_steps = 1
    for _ in range(num_calibration_steps):
        specs = []
        for i in range(64):
            index = np.random.randint(
                low=0,
                high=len(all_data) - 1,)
            spec = all_data[index][0]
            spec = np.repeat(spec[..., np.newaxis], 3, -1)
            specs.append(spec)
        # Get sample input data as a numpy array in a method of your choosing.
        # print(len(specs))
        yield [specs]


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


def main():

    train_file = '/Users/alirachidi/Documents/Sonavi_Labs/ObjectDetection/data/datasets/all_sw_log_preprocessed_v2_param_v2_8000.pkl'
    file_stream = file_io.FileIO(train_file, mode="rb")
    data = pickle.load(file_stream)
    convert = False
    interpret = True
    saved_model_dir = '/Users/alirachidi/Documents/Sonavi_Labs/ObjectDetection/data/saved_model/'
    tflite_dir = saved_model_dir + 'second_saved_model.tflite'
    if convert:
        all_data = [data[i][y][z] for i in range(len(data)) for y in range(
            len(data[i])) for z in range(len(data[i][y]))]
        np.random.shuffle(all_data)
        print("Length of Val Set:".format(len(all_data)))
        class_acc = class_accuracy()
        model = load_model(saved_model_dir + "model.h5", custom_objects={
                           "class_accuracy": class_acc})
        print(model.summary())
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        print(converter)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen(all_data)
        tflite_quant_model = converter.convert()
        print("Model converted.")

        print("TFLite mode: {}".format(tflite_dir))
        with open(tflite_dir, 'wb') as f:
            print("Writing TF Lite model to directory.")
            f.write(tflite_quant_model)
        write_to_file(tflite_quant_model, saved_model_dir + 'model.pkl')

    if interpret:
        validation_data = data[1][0] + data[1][1] + data[1][2] + data[1][3]
        np.random.shuffle(validation_data)
        print("Length of Val Set:".format(len(validation_data)))
        interpreter = tf.lite.Interpreter(model_path=tflite_dir)
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

        pool_size = 5
        p = multiprocessing.Pool(pool_size)
        func = partial(predict, interpreter, input_details, output_details)

        # print(len(validation_data))
        # print(type(validation_data[0][0]))
        # for i, sample in enumerate(validation_data):
        #     validation_data[i] = validation_data[i][0]
        # print(validation_data[0])
        validation_data = [1, 35, 345, 2]
        p.map(func, validation_data)

        # for sample in items:
        #     pool.apply_async(worker, (item,))

        p.close()
        p.join()

        # executor = concurrent.futures.ProcessPoolExecutor(10)
        # futures = []
        # for sample in validation_data:
        #     future = executor.submit(predict, sample, interpreter)
        #     print(future.result())
        #     futures.append(future)
        # concurrent.futures.wait(futures)


# total_correct = 0
# for i in range(len(validation_data)):
#     start_time = time.time()
#     input_data = validation_data[i][0]
#     input_data = np.repeat(input_data[..., np.newaxis], 3, -1)
#     input_data = np.reshape(input_data, (1,) + input_data.shape)
#     # print(input_data.shape)
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     label = [validation_data[i][1], validation_data[i][2]]
#     print("Label: {}".format(label))
#     print("Output: {}".format(output_data))
#     correct_pred = is_correct(label, output_data)
#     if correct_pred == True:
#         total_correct += 1
#     print(" Acc: %.0f " % ((total_correct/(i+1))*100))
#     print("--- %.2f seconds ---" % (time.time() - start_time))
#     print("------------")
# if (i == 100):
#     print("Excepted time left: {}")
# print("Final Accuracy: {}".format(total_correct/len(validation_data)))
print("Interpreter Invoked.")


if __name__ == "__main__":
    main()
