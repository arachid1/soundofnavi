import tensorflow as tf

class class_accuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super(class_accuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name="acc", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # tf.print("start")
        # tf.print(y_true)
        # tf.print(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.round(y_pred), tf.int32)
        # tf.print("new y_pred")
        # tf.print(y_pred)
        matches = tf.equal(y_true, y_pred)
        # tf.print(matches)
        matches = tf.map_fn(
            lambda x: tf.equal(tf.reduce_sum(
                tf.cast(x, tf.float32)), 2), matches
        )
        # tf.print(matches)
        matches = tf.cast(matches, tf.float32)
        acc = tf.reduce_sum(matches) / \
            tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        # tf.print(acc)
        self.accuracy.assign(acc)

    def result(self):
        tf.print(self.accuracy)
        return self.accuracy


# class confusion_matrix(tf.keras.metrics.Metrics):

#     def __init__(self, **kwargs):
#         # Initialise as normal and add flag variable for when to run computation
#         super(MyCustomMetric, self).__init__(**kwargs)
#         self.cm = self.add_weight(name='cm', initializer='zeros')
#         self.update_metric = tf.Variable(False)

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # Use conditional to determine if computation is done
#         if self.update_metric:
#             y_true = tf.map_fn(
#                 lambda x: tf.equal(tf.reduce_sum(
#                     tf.cast(x, tf.float32)), 2), y_true
#             )
#             # run computation
#             # self.cmx

#     def result(self):
#         return self.cm

#     def reset_states(self):
#         self.cm.assign(0.)


def calc_accuracy(y_true, y_pred):

    # tf.print("start") 
    # tf.print(tf.shape(y_true))
    # tf.print(tf.shape(y_pred))
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(tf.round(y_pred), tf.int32)
    # tf.print(y_pred)
    # tf.print("new y_pred")
    # tf.print(y_pred)
    matches = tf.equal(y_true, y_pred)
    # tf.print(matches)
    # tf.print(matches)
    matches = tf.map_fn(
        lambda x: tf.equal(tf.reduce_sum(
            tf.cast(x, tf.float32)), 2), matches
    )
    # tf.print(matches)
    matches = tf.cast(matches, tf.float32)
    # tf.print(matches)
    acc = tf.reduce_sum(matches) / \
        tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    # tf.print(acc)
    # tf.print('\n')
    # tf.print(acc)
    return acc
    # self.accuracy.assign(acc)

