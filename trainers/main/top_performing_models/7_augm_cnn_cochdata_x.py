KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
PADDING = "same"
CHANNELS = 32
DROPOUT = 0.1
DENSE_LAYER = 32
i = layers.Input(shape=SHAPE, batch_size=BATCH_SIZE)
x = layers.BatchNormalization()(i)
x = layers.ReLU()(x)
x = layers.Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, activation="relu")(x)
x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
CHANNELS = CHANNELS*2
x = layers.Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, activation="relu")(x)
x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
CHANNELS = CHANNELS*2
x = layers.Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, activation="relu")(x)
x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
CHANNELS = CHANNELS*2
x = layers.Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, activation="relu")(x)
x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Flatten()(x)
x = layers.Dense(DENSE_LAYER)(x)
x = layers.Dropout(DROPOUT)(x)
o = layers.Dense(N_CLASSES, activity_regularizer=l2(
    LL2_REG), activation="sigmoid")(x)