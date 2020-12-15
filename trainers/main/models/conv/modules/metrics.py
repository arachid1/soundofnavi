import tensorflow as tf

class class_accuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
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