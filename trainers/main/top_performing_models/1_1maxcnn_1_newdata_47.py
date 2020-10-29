pooled_outputs = []
filter_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45]
time_length = SHAPE[1]
freq_length = SHAPE[0]
num_filters = 64
# tf.print("SHAPE")
# tf.print(SHAPE)
i = layers.Input(shape=SHAPE,
                 batch_size=BATCH_SIZE, name="input")
for index, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        filter_shape = (freq_length, filter_size)
        # tf.print("filter_shape")
        # tf.print(filter_shape)
        # filter_shape = [filter_size, freq_length, 1, num_filters]
        # W = tf.Variable(tf.random.truncated_normal(
        #     filter_shape, stddev=0.1), name="W")
        # print(W)
        # tf.print(W)
        b = tf.Variable(tf.constant(
            0.1, shape=[num_filters]), name="b")
        x = layers.Conv2D(
            num_filters, kernel_size=filter_shape, strides=[1, 1], padding="VALID")(i)
        x = layers.ReLU()(tf.nn.bias_add(x, b))
        # tf.print("x.shape")
        # tf.print(x.shape)
        x = layers.MaxPooling2D(
            pool_size=(1, time_length - filter_size + 1), strides=[1, 1], padding='VALID')(x)
        # tf.print("x.shape after maxpool2D")
        # print(x[0])
        # tf.print(x.shape)
        pooled_outputs.append(x)
        #pooled_outputs = layers.Concatenate()([pooled_outputs, x])

        # tf.print("pooled_outputs")
        # print(pooled_outputs)

intermediate_output = pooled_outputs

num_filters_total = num_filters * len(filter_sizes)
#h_pool = tf.concat(3, intermediate_output)
#h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
h_pool_flat = tf.reshape(intermediate_output, [-1, num_filters_total])

x = layers.Flatten()(h_pool_flat)
# o = layers.Dense(1024)(x)
o = layers.Dense(N_CLASSES, activity_regularizer=l2(
    LL2_REG), activation="sigmoid")(x)
