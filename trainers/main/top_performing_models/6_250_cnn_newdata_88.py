KERNEL_SIZE = (6, 6)
POOL_SIZE = (2, 2)
i = layers.Input(shape=SHAPE, batch_size=BATCH_SIZE)
x = layers.BatchNormalization()(i)
x = layers.Conv2D(
    16, kernel_size=(1, 1), activation="relu", padding="same",
)(x)
x = layers.Conv2D(
    32, kernel_size=KERNEL_SIZE, activation="relu", padding="same",
)(x)
x = layers.Conv2D(64, kernel_size=(
    1, 1), activation="relu", padding="same",)(x)
x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same", )(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(
    64, kernel_size=KERNEL_SIZE, activation="relu", padding="same",
)(x)
x = layers.Conv2D(128, kernel_size=(
    1, 1), activation="relu", padding="same",)(x)
x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same")(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(
    128, kernel_size=KERNEL_SIZE, activation="relu", padding="same",
)(x)
x = layers.Conv2D(256, kernel_size=(
    1, 1), activation="relu", padding="same",)(x)
x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same",)(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(
    256, kernel_size=KERNEL_SIZE, activation="relu", padding="same",
)(x)
x = layers.Conv2D(512, kernel_size=(
    1, 1), activation="relu", padding="same",)(x)
x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same", )(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(
    512, kernel_size=KERNEL_SIZE, activation="relu", padding="same",
)(x)
x = layers.Conv2D(1024, kernel_size=(
    1, 1), activation="relu", padding="same",)(x)
x = layers.AveragePooling2D(pool_size=POOL_SIZE, padding="same",)(x)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(1024, kernel_size=(
    1, 1), activation="relu", padding="same",)(x)
x = layers.Conv2D(2048, kernel_size=KERNEL_SIZE,
                  activation="relu", padding="same",)(x)
x = layers.Flatten()(x)
o = layers.Dense(N_CLASSES, activity_regularizer=l2(
    LL2_REG), activation="sigmoid")(x)
