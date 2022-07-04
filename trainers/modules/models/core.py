import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, AveragePooling2D, InputSpec, TimeDistributed
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.initializers import RandomUniform, RandomNormal
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils
from ..main import parameters

class Distiller(Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
    
    def call(self, inputs: tf.Tensor, training: bool = True):
        # teacher_predictions = self.teacher(inputs, training=False)
        student_predictions = self.student(inputs, training=training)
        return student_predictions

    def train_step(self, inputs):
        # Unpack data
        x, y = inputs

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)
        # teacher_predictions = self(x)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            distillation_loss = self.distillation_loss_fn(
                tf.nn.sigmoid(teacher_predictions / self.temperature),
                tf.nn.sigmoid(student_predictions / self.temperature),
            )
            # distillation_loss = self.distillation_loss_fn(
            #     teacher_predictions,
            #     student_predictions,
            # )
            # tf.print("here")
            # tf.print(student_loss)
            # tf.print(distillation_loss)
            # tf.print(teacher_predictions)
            # tf.print(student_predictions)
            # tf.print(tf.nn.sigmoid(teacher_predictions))
            # tf.print(tf.nn.sigmoid(student_predictions))

            if parameters.distill_features:
                mel_spec = self.teacher(x, training=False, return_spec=True)
                leaf_spec = self.student(x, training=False, return_spec=True)
                features_loss = tf.reduce_mean(tf.math.square(mel_spec-leaf_spec))
                distillation_loss = distillation_loss + features_loss

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

    def return_models(self):
        return self.teacher, self.student


class _ConvMoE(Layer):
    """Abstract nD convolution layer mixture of experts (private, used as implementation base).
    """

    def __init__(self, rank,
                 n_filters,
                 n_experts_per_filter,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 expert_activation=None,
                 gating_activation=None,
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_kernel_initializer_scale=1.0,
                 gating_kernel_initializer_scale=1.0,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(_ConvMoE, self).__init__(**kwargs)
        self.rank = rank
        self.n_filters = n_filters
        self.n_experts_per_filter = n_experts_per_filter
        self.n_total_filters = self.n_filters * self.n_experts_per_filter
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')

        self.expert_activation = activations.get(expert_activation)
        self.gating_activation = activations.get(gating_activation)

        self.use_expert_bias = use_expert_bias
        self.use_gating_bias = use_gating_bias

        self.expert_kernel_initializer_scale = expert_kernel_initializer_scale
        self.gating_kernel_initializer_scale = gating_kernel_initializer_scale

        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gating_bias_initializer = initializers.get(gating_bias_initializer)

        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gating_kernel_regularizer = regularizers.get(gating_kernel_regularizer)

        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gating_bias_regularizer = regularizers.get(gating_bias_regularizer)

        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gating_kernel_constraint = constraints.get(gating_kernel_constraint)

        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gating_bias_constraint = constraints.get(gating_bias_constraint)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        expert_init_std = self.expert_kernel_initializer_scale / np.sqrt(input_dim*np.prod(self.kernel_size))
        gating_init_std = self.gating_kernel_initializer_scale / np.sqrt(np.prod(input_shape[1:]))

        expert_kernel_shape = self.kernel_size + (input_dim, self.n_total_filters)
        self.expert_kernel = self.add_weight(shape=expert_kernel_shape,
                                      initializer=RandomNormal(mean=0., stddev=expert_init_std),
                                      name='expert_kernel',
                                      regularizer=self.expert_kernel_regularizer,
                                      constraint=self.expert_kernel_constraint)

        gating_kernel_shape = input_shape[1:] + (self.n_filters, self.n_experts_per_filter)
        self.gating_kernel = self.add_weight(shape=gating_kernel_shape,
                                      initializer=RandomNormal(mean=0., stddev=gating_init_std),
                                      name='gating_kernel',
                                      regularizer=self.gating_kernel_regularizer,
                                      constraint=self.gating_kernel_constraint)

        if self.use_expert_bias:

            expert_bias_shape = ()
            for i in range(self.rank):
                expert_bias_shape = expert_bias_shape + (1,)
            expert_bias_shape = expert_bias_shape + (self.n_filters, self.n_experts_per_filter)

            self.expert_bias = self.add_weight(shape=expert_bias_shape,
                                        initializer=self.expert_bias_initializer,
                                        name='expert_bias',
                                        regularizer=self.expert_bias_regularizer,
                                        constraint=self.expert_bias_constraint)
        else:
            self.expert_bias = None

        if self.use_gating_bias:
            self.gating_bias = self.add_weight(shape=(self.n_filters, self.n_experts_per_filter),
                                        initializer=self.gating_bias_initializer,
                                        name='gating_bias',
                                        regularizer=self.gating_bias_regularizer,
                                        constraint=self.gating_bias_constraint)
        else:
            self.gating_bias = None

        self.o_shape = self.compute_output_shape(input_shape=input_shape)
        self.new_gating_outputs_shape = (-1,)
        for i in range(self.rank):
            self.new_gating_outputs_shape = self.new_gating_outputs_shape + (1,)
        self.new_gating_outputs_shape = self.new_gating_outputs_shape + (self.n_filters, self.n_experts_per_filter)

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.rank == 1:
            expert_outputs = K.conv1d(
                inputs,
                self.expert_kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            expert_outputs = K.conv2d(
                inputs,
                self.expert_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            expert_outputs = K.conv3d(
                inputs,
                self.expert_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        expert_outputs = K.reshape(expert_outputs, (-1,) + self.o_shape[1:-1] + (self.n_filters, self.n_experts_per_filter))

        if self.use_expert_bias:
            expert_outputs = K.bias_add(
                expert_outputs,
                self.expert_bias,
                data_format=self.data_format)

        if self.expert_activation is not None:
            expert_outputs = self.expert_activation(expert_outputs)

        gating_outputs = tf.tensordot(inputs, self.gating_kernel, axes=self.rank+1) # samples x n_filters x n_experts_per_filter

        if self.use_gating_bias:
            gating_outputs = K.bias_add(
                gating_outputs,
                self.gating_bias,
                data_format=self.data_format)

        if self.gating_activation is not None:
            gating_outputs = self.gating_activation(gating_outputs)

        gating_outputs = K.reshape(gating_outputs, self.new_gating_outputs_shape)
        outputs = K.sum(expert_outputs * gating_outputs, axis=-1, keepdims=False)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.n_filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.n_filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'n_filters': self.n_filters,
            'n_experts_per_filter': self.n_experts_per_filter,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'expert_activation': activations.serialize(self.expert_activation),
            'gating_activation': activations.serialize(self.gating_activation),
            'use_expert_bias': self.use_expert_bias,
            'use_gating_bias': self.use_gating_bias,
            'expert_kernel_initializer_scale':self.expert_kernel_initializer_scale,
            'gating_kernel_initializer_scale':self.gating_kernel_initializer_scale,
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gating_bias_initializer': initializers.serialize(self.gating_bias_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gating_kernel_regularizer': regularizers.serialize(self.gating_kernel_regularizer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gating_bias_regularizer': regularizers.serialize(self.gating_bias_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gating_kernel_constraint': constraints.serialize(self.gating_kernel_constraint),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gating_bias_constraint': constraints.serialize(self.gating_bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(_ConvMoE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Conv2DMoE(_ConvMoE):
    """2D convolution layer (e.g. spatial convolution over images).
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)`
        if `data_format` is `"channels_last"`.
    # Output shape
        4D tensor with shape:
        `(samples, n_filters, new_rows, new_cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, n_filters)`
        if `data_format` is `"channels_last"`.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 n_filters,
                 n_experts_per_filter,
                 kernel_size,
                 strides=(1,1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1,1),
                 expert_activation=None,
                 gating_activation=None,
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_kernel_initializer_scale=1.0,
                 gating_kernel_initializer_scale=1.0,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Conv2DMoE, self).__init__(
            rank=2,
            n_filters=n_filters,
            n_experts_per_filter=n_experts_per_filter,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            expert_activation=expert_activation,
            gating_activation=gating_activation,
            use_expert_bias=use_expert_bias,
            use_gating_bias=use_gating_bias,
            expert_kernel_initializer_scale=expert_kernel_initializer_scale,
            gating_kernel_initializer_scale=gating_kernel_initializer_scale,
            expert_bias_initializer=expert_bias_initializer,
            gating_bias_initializer=gating_bias_initializer,
            expert_kernel_regularizer=expert_kernel_regularizer,
            gating_kernel_regularizer=gating_kernel_regularizer,
            expert_bias_regularizer=expert_bias_regularizer,
            gating_bias_regularizer=gating_bias_regularizer,
            expert_kernel_constraint=expert_kernel_constraint,
            gating_kernel_constraint=gating_kernel_constraint,
            expert_bias_constraint=expert_bias_constraint,
            gating_bias_constraint=gating_bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def get_config(self):
        config = super(Conv2DMoE, self).get_config()
        config.pop('rank')
        return config

class DenseMoE(Layer):
    """Mixture-of-experts layer.
    Implements: y = sum_{k=1}^K g(v_k * x) f(W_k * x)
        # Arguments
        units: Positive integer, dimensionality of the output space.
        n_experts: Positive integer, number of experts (K).
        expert_activation: Activation function for the expert model (f).
        gating_activation: Activation function for the gating model (g).
        use_expert_bias: Boolean, whether to use biases in the expert model.
        use_gating_bias: Boolean, whether to use biases in the gating model.
        expert_kernel_initializer_scale: Float, scale of Glorot uniform initialization for expert model weights.
        gating_kernel_initializer_scale: Float, scale of Glorot uniform initialization for gating model weights.
        expert_bias_initializer: Initializer for the expert biases.
        gating_bias_initializer: Initializer fot the gating biases.
        expert_kernel_regularizer: Regularizer for the expert model weights.
        gating_kernel_regularizer: Regularizer for the gating model weights.
        expert_bias_regularizer: Regularizer for the expert model biases.
        gating_bias_regularizer: Regularizer for the gating model biases.
        expert_kernel_constraint: Constraints for the expert model weights.
        gating_kernel_constraint: Constraints for the gating model weights.
        expert_bias_constraint: Constraints for the expert model biases.
        gating_bias_constraint: Constraints for the gating model biases.
        activity_regularizer: Activity regularizer.
    # Input shape
        nD tensor with shape: (batch_size, ..., input_dim).
        The most common situation would be a 2D input with shape (batch_size, input_dim).
    # Output shape
        nD tensor with shape: (batch_size, ..., units).
        For example, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).
    """
    def __init__(self, units,
                 n_experts,
                 expert_activation=None,
                 gating_activation=None,
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_kernel_initializer_scale=1.0,
                 gating_kernel_initializer_scale=1.0,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseMoE, self).__init__(**kwargs)
        self.units = units
        self.n_experts = n_experts

        self.expert_activation = activations.get(expert_activation)
        self.gating_activation = activations.get(gating_activation)

        self.use_expert_bias = use_expert_bias
        self.use_gating_bias = use_gating_bias

        self.expert_kernel_initializer_scale = expert_kernel_initializer_scale
        self.gating_kernel_initializer_scale = gating_kernel_initializer_scale

        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gating_bias_initializer = initializers.get(gating_bias_initializer)

        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gating_kernel_regularizer = regularizers.get(gating_kernel_regularizer)

        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gating_bias_regularizer = regularizers.get(gating_bias_regularizer)

        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gating_kernel_constraint = constraints.get(gating_kernel_constraint)

        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gating_bias_constraint = constraints.get(gating_bias_constraint)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        expert_init_lim = np.sqrt(3.0*self.expert_kernel_initializer_scale / (max(1., float(input_dim + self.units) / 2)))
        gating_init_lim = np.sqrt(3.0*self.gating_kernel_initializer_scale / (max(1., float(input_dim + 1) / 2)))


        # print((input_dim, self.units, self.n_experts))
        self.expert_kernel = self.add_weight(shape=(input_dim, self.units, self.n_experts),
                                      initializer=RandomUniform(minval=-expert_init_lim,maxval=expert_init_lim),
                                      name='expert_kernel',
                                      regularizer=self.expert_kernel_regularizer,
                                      constraint=self.expert_kernel_constraint)

        # print((input_dim, self.n_experts))
        self.gating_kernel = self.add_weight(shape=(input_dim, self.n_experts),
                                      initializer=RandomUniform(minval=-gating_init_lim,maxval=gating_init_lim),
                                      name='gating_kernel',
                                      regularizer=self.gating_kernel_regularizer,
                                      constraint=self.gating_kernel_constraint)
        # tf.print(self.gat)

        if self.use_expert_bias:
            self.expert_bias = self.add_weight(shape=(self.units, self.n_experts),
                                        initializer=self.expert_bias_initializer,
                                        name='expert_bias',
                                        regularizer=self.expert_bias_regularizer,
                                        constraint=self.expert_bias_constraint)
        else:
            self.expert_bias = None

        if self.use_gating_bias:
            self.gating_bias = self.add_weight(shape=(self.n_experts,),
                                        initializer=self.gating_bias_initializer,
                                        name='gating_bias',
                                        regularizer=self.gating_bias_regularizer,
                                        constraint=self.gating_bias_constraint)
        else:
            self.gating_bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):


        expert_outputs = tf.tensordot(inputs, self.expert_kernel, axes=1)
        # tf.print(tf.shape(expert_outputs))
        # if self.use_expert_bias:
        #     expert_outputs = K.bias_add(expert_outputs, self.expert_bias)
        if self.expert_activation is not None:
            expert_outputs = self.expert_activation(expert_outputs)

        gating_outputs = K.dot(inputs, self.gating_kernel)
        # tf.print(tf.shape(gating_outputs))
        # if self.use_gating_bias:
        #     gating_outputs = K.bias_add(gating_outputs, self.gating_bias)
        if self.gating_activation is not None:
            gating_outputs = self.gating_activation(gating_outputs)
        # tf.print(tf.shape(gating_outputs))

        output = K.sum(expert_outputs * K.repeat_elements(K.expand_dims(gating_outputs, axis=1), self.units, axis=1), axis=2)
        # tf.print(tf.shape(output))
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'n_experts':self.n_experts,
            'expert_activation': activations.serialize(self.expert_activation),
            'gating_activation': activations.serialize(self.gating_activation),
            'use_expert_bias': self.use_expert_bias,
            'use_gating_bias': self.use_gating_bias,
            'expert_kernel_initializer_scale': self.expert_kernel_initializer_scale,
            'gating_kernel_initializer_scale': self.gating_kernel_initializer_scale,
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gating_bias_initializer': initializers.serialize(self.gating_bias_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gating_kernel_regularizer': regularizers.serialize(self.gating_kernel_regularizer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gating_bias_regularizer': regularizers.serialize(self.gating_bias_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gating_kernel_constraint': constraints.serialize(self.gating_kernel_constraint),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gating_bias_constraint': constraints.serialize(self.gating_bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(DenseMoE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MixDepthGroupConvolution2D(tf.keras.layers.Layer):
    def __init__(self, kernels=[3, 5],
                 conv_kwargs=None,
                 **kwargs):
        super(MixDepthGroupConvolution2D, self).__init__(**kwargs)

        if conv_kwargs is None:
            conv_kwargs = {
                'strides': (1, 1),
                'padding': 'same',
                'dilation_rate': (1, 1),
                'use_bias': False,
            }
        self.channel_axis = -1 
        self.kernels = kernels
        self.groups = len(self.kernels)
        self.strides = conv_kwargs.get('strides', (1, 1))
        self.padding = conv_kwargs.get('padding', 'same')
        self.dilation_rate = conv_kwargs.get('dilation_rate', (1, 1))
        self.use_bias = conv_kwargs.get('use_bias', False)
        self.conv_kwargs = conv_kwargs or {}

        self.layers = [tf.keras.layers.DepthwiseConv2D(kernels[i],
                                       strides=self.strides,
                                       padding=self.padding,
                                       activation=tf.nn.relu,                
                                       dilation_rate=self.dilation_rate,
                                       use_bias=self.use_bias,
                                       kernel_initializer='he_normal')
                        for i in range(self.groups)]

    def call(self, inputs, **kwargs):
        if len(self.layers) == 1:
            return self.layers[0](inputs)
        # tf.print("here")
        filters = K.int_shape(inputs)[self.channel_axis]
        # tf.print(tf.shape(filters))
        # tf.print("here2")
        splits  = self.split_channels(filters, self.groups)
        # tf.print(tf.shape(splits))
        # tf.print("here3")
        x_splits  = tf.split(inputs, splits, self.channel_axis)
        # tf.print(tf.shape(x_splits))
        # tf.print("here4")
        # x_outputs = []
        # for x, c in zip(x_splits, self.layers):
        #     # print(tf.shape(x))
        #     # tf.print(tf.shape(c(x)))
        #     x_outputs.append(c(x))
        x_outputs = [c(x) for x, c in zip(x_splits, self.layers)]
        # tf.print(tf.shape(x_outputs))
        # tf.print("here5")
        return tf.keras.layers.concatenate(x_outputs, 
                                           axis=self.channel_axis)

    def split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def get_config(self):
        config = {
            'kernels': self.kernels,
            'groups': self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'conv_kwargs': self.conv_kwargs
        }
        base_config = super(MixDepthGroupConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
        

class InvertedResidual(Layer):
    def __init__(self, filters, strides, activation=ReLU(), kernel_size=3, expansion_factor=6, padding="valid",
                 regularizer=None, trainable=True, name=None, **kwargs):
        super(InvertedResidual, self).__init__(
            trainable=trainable, name=name, **kwargs)
        self.filters = filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.activation = activation
        self.regularizer = regularizer
        self.padding = padding
        self.channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    def build(self, input_shape):
        input_channels = int(input_shape[self.channel_axis])  # C
        # tf.print("self.regularizer")
        # tf.print(self.regularizer)
        self.ptwise_conv1 = Conv2D(filters=int(input_channels*self.expansion_factor), padding=self.padding,
                                   kernel_size=1, kernel_regularizer=None, use_bias=False)
        self.dwise = DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                     kernel_regularizer=None, use_bias=False)
        self.ptwise_conv2 = Conv2D(filters=self.filters, kernel_size=1, padding=self.padding,
                                   kernel_regularizer=None, use_bias=False)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

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
        # tf.print("start")
        # tf.print(input_x.shape[1:])
        # tf.print(x.shape[1:])
        if input_x.shape[1:] == x.shape[1:]:
            # tf.print("here")
            x += input_x
        return x

    def get_config(self):
        cfg = super(InvertedResidual, self).get_config()
        cfg.update({'filters': self.filters,
                    'strides': self.strides,
                    'padding': self.padding,
                    'regularizer': self.strides,
                    'expansion_factor': self.expansion_factor,
                    'activation': self.activation})
        return cfg

    def compute_output_shape(self, input_shape):
        tf.print(input_shape)
        return (input_shape[0], input_shape[1], self.filters)

class TimeDistributedInvertedResidual(Layer):
    def __init__(self, filters, strides, activation=ReLU(), kernel_size=3, expansion_factor=6,
                 regularizer=None, trainable=True, name=None, **kwargs):
        super(TimeDistributedInvertedResidual, self).__init__(
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
        # tf.print("self.regularizer")
        # tf.print(self.regularizer)
        self.ptwise_conv1 = TimeDistributed(Conv2D(filters=int(input_channels*self.expansion_factor),
                                   kernel_size=1, kernel_regularizer=None, use_bias=False))
        self.dwise = TimeDistributed(DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.strides,
                                     kernel_regularizer=None, padding='same', use_bias=False))
        self.ptwise_conv2 = TimeDistributed(Conv2D(filters=self.filters, kernel_size=1,
                                   kernel_regularizer=None, use_bias=False))
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

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
        # tf.print("start")
        # tf.print(input_x.shape[1:])
        # tf.print(x.shape[1:])
        if input_x.shape[1:] == x.shape[1:]:
            # tf.print("here")
            x += input_x
        return x

    def get_config(self):
        cfg = super(TimeDistributedInvertedResidual, self).get_config()
        cfg.update({'filters': self.filters,
                    'strides': self.strides,
                    'regularizer': self.strides,
                    'expansion_factor': self.expansion_factor,
                    'activation': self.activation})
        return cfg

    def compute_output_shape(self, input_shape):
        tf.print(input_shape)
        return (input_shape[0], input_shape[1], self.filters)

def create_bn_act_1_conv_pool(INPUT, KERNEL_SIZE, POOL_SIZE, CHANNELS, PADDING):
    x = BatchNormalization()(INPUT)
    x = ReLU()(x)
    x = Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, activation="relu")(x)
    x = AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
    return x

def create_bn_act_3_convs_pool(INPUT, KERNEL_SIZE, POOL_SIZE, CHANNELS, PADDING):
    x = BatchNormalization()(INPUT)
    x = ReLU()(x)
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(x)
    x = Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, activation="relu")(x)
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(x)
    x = AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
    return x

def create_bn_act_3_strided_convs(INPUT, KERNEL_SIZE, POOL_SIZE, CHANNELS, PADDING):
    x = BatchNormalization()(INPUT)
    x = ReLU()(x)
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(x)
    x = Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, strides=2, padding=PADDING, activation="relu")(x)
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(x)
    x = AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
    return x
    
def create_3_convs_relu_avg_bn(INPUT, KERNEL_SIZE, POOL_SIZE, CHANNELS, PADDING):
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(INPUT)
    x = Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, activation="relu")(x)
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(x)
    x = AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
    x = BatchNormalization()(x)
    return x
    