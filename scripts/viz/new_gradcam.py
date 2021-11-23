import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.python.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras import activations
import tensorflow.keras.backend as K
# Display
from IPython.display import Image, display
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import librosa
from librosa import display as dis

class InvertedResidual(layers.Layer):
    def __init__(self, filters, strides, activation=ReLU(), kernel_size=3, expansion_factor=6,
                 regularizer=None, trainable=True, name=None, **kwargs):
        super(InvertedResidual, self).__init__(
            trainable=trainable, name=name, **kwargs)
        self.filters = filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.activation = activation
        self.regularizer = regularizer
        self.channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    def build(self, input_shape):
        input_channels = int(input_shape[self.channel_axis])  # C
        self.ptwise_conv1 = Conv2D(filters=int(input_channels*self.expansion_factor),
                                   kernel_size=1, kernel_regularizer=self.regularizer, use_bias=False)
        self.dwise = DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.strides,
                                     kernel_regularizer=self.regularizer, padding='same', use_bias=False)
        self.ptwise_conv2 = Conv2D(filters=self.filters, kernel_size=1,
                                   kernel_regularizer=self.regularizer, use_bias=False)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        
        # input = C 
        # 1x1 -> 4C
        # depthwise 3x3 -> 4C
        # 1x1 -> C 
        # ouput = C

    def call(self, input_x, training=False):
        # Expansion
        x = self.ptwise_conv1(input_x)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        # Spatial filtering
        x = self.dwise(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        # back to low-channels w/o activation
        x = self.ptwise_conv2(x)
        x = self.bn3(x, training=training)
        # Residual connection only if i/o have same spatial and depth dims
        if input_x.shape[1:] == x.shape[1:]:
            x += input_x
        return x

    def get_config(self):
        cfg = super(InvertedResidual, self).get_config()
        cfg.update({'filters': self.filters,
                    'strides': self.strides,
                    'regularizer': self.regularizer,
                    'expansion_factor': self.expansion_factor,
                    'activation': self.activation})
        return cfg

def return_mod10(SHAPE, BATCH_SIZE, N_CLASSES):
    KERNEL_SIZE = (3, 3)
    POOL_SIZE = (2, 2)
    PADDING = "same"
    CHANNELS = 32
    DROPOUT = 0.1
    DENSE_LAYER = 32
    LL2_REG = 0
    i = layers.Input(shape=SHAPE)
    x = layers.BatchNormalization()(i)
    tower_1 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(8, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(8, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(8, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    tower_1 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(16, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
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
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)
    model = Model(inputs=i, outputs=o, name="conv2d")
    return model

def return_mod9(SHAPE, BATCH_SIZE, N_CLASSES):

    KERNEL_SIZE = 6
    POOL_SIZE = (2, 2)
    LL2_REG = 0
    i = layers.Input(shape=SHAPE,)
    x = layers.BatchNormalization()(i)
    tower_1 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
    tower_2 = layers.Conv2D(16, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    tower_3 = layers.Conv2D(16, (1,1), padding='same', activation='relu')(tower_3)
    x = layers.Concatenate(axis=3)([tower_1, tower_2, tower_3])
    x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = InvertedResidual(filters=32, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = InvertedResidual(filters=64, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = InvertedResidual(filters=128, strides=1, kernel_size=KERNEL_SIZE,)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = InvertedResidual(filters=256, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = layers.AveragePooling2D(pool_size=POOL_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = InvertedResidual(filters=512, strides=1, kernel_size=KERNEL_SIZE)(x)
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(N_CLASSES, activity_regularizer=l2(
        LL2_REG), activation="sigmoid")(x)

    model = Model(inputs=i, outputs=o, name="conv2d")
    
    return model

def save_original(spec, sr, dest):
    fig = plt.figure(figsize=(20, 10))
    dis.specshow(
        spec,
        # y_axis="log",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar()
    plt.show()
    plt.savefig(os.path.join(dest, "orig"))

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    # img = keras.preprocessing.image.load_img(img_path)
    # img = keras.preprocessing.image.img_to_array(img)
    print("inside save")
    print(img.shape)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    print(jet_heatmap.size)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    print(jet_heatmap.size)
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    print(jet_heatmap.size)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # flip it
    superimposed_img = superimposed_img.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def main():

    SHAPE = (128, 1250, 3)
    BATCH_SIZE = 1
    N_CLASSES = 1

    version = 'v29'
    filenames = ["01-95637_170819-001-1_F_1", "01-95637_170819-001-2_F_1", "01-95637_170819-001-3_F_1", "01-95637_170819-001-4_F_1", "01-95637_170819-001-5_F_1", "01-95637_170819-001-6_F_1"]
    model_path = "/home/alirachidi/classification_algorithm/cache/conv___04_0121___all_sw_coch_preprocessed_v2_param_v29_augm_v0_cleaned_8000___test___2049/third.h5"

    # MODEL
    model = return_mod9(SHAPE, BATCH_SIZE, N_CLASSES)
    # model = return_mod10(SHAPE, BATCH_SIZE, N_CLASSES)    
    model.load_weights(model_path)
    model.summary()

    for filename in filenames:

        file_path = '../../data/txt_datasets/all_sw_coch_preprocessed_v2_param_{}_augm_v0_cleaned_8000/{}.txt'.format(version, filename)
        dest = os.path.join("viz_outputs", "pneumonia", filename)

        if not os.path.exists(dest):
            os.mkdir(dest)

        # SPEC
        spec = np.loadtxt(file_path, delimiter=',')
        save_original(spec, 8000, dest)
        spec = np.repeat(spec[..., np.newaxis], 3, -1)
    
        preds = model.predict(np.array([spec]))

        with open(os.path.join(dest, "info.txt"), "w") as f:
            f.write("Predicted: {}".format(preds))
        
        print("Predicted: {}".format(preds))

        model.layers[-1].activation = None # TODO: CHECK OTHER LAYERS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        last_conv_layer_name = "inverted_residual"

        for i in range(1, 11):
            print("Generating gradcam for {}...".format(last_conv_layer_name))

            # heatmap = make_gradcam_heatmap(np.array([spec]), model, last_conv_layer_name, pred_index=0)

            # save_and_display_gradcam(spec, heatmap, cam_path=os.path.join(dest, "cam_crackles_{}.jpg".format(last_conv_layer_name)))

            # heatmap = make_gradcam_heatmap(np.array([spec]), model, last_conv_layer_name, pred_index=1)

            # save_and_display_gradcam(spec, heatmap, cam_path=os.path.join(dest, "cam_wheezes_{}.jpg".format(last_conv_layer_name)))

            heatmap = make_gradcam_heatmap(np.array([spec]), model, last_conv_layer_name, pred_index=0)

            save_and_display_gradcam(spec, heatmap, cam_path=os.path.join(dest, "cam_{}.jpg".format(last_conv_layer_name)))

            last_conv_layer_name = "inverted_residual_{}".format(i)

    


if __name__ == "__main__":
    main()