import tensorflow as tf

class pneumonia_confusion_matrix(tf.keras.metrics.Metric):

    def __init__(self, name='confusion_matrix', **kwargs):
        super(pneumonia_confusion_matrix, self).__init__(name=name, **kwargs)
        self.cm = self.add_weight(name='cm', initializer='zeros')
            
    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.round(y_pred), tf.int32)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        new_cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred)
        tf.print(new_cm)
        self.cm.assign_add(new_cm)
        
    def result(self):
        return self.cm
    
    # def generate_confusion_matrix(self, y_true, y_pred):

    #     y_true = tf.cast(y_true, tf.int32)
    #     y_pred = tf.cast(tf.round(y_pred), tf.int32)

    #     y_true = tf.reshape(y_true, [-1])
    #     y_pred = tf.reshape(y_pred, [-1])
    #     cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred, num_classes=2)
    #     # tf.print(cm)
    #     # if (tf.size(y_true) >= 32):
    #     #     tf.print(cm)
    #     return cm
    
    # def process_confusion_matrix(self):
    #     "returns precision, recall and f1 along with overall accuracy"
    #     cm=self.total_cm
    #     diag_part=tf.linalg.diag_part(cm)
    #     precision=diag_part/(tf.reduce_sum(cm,0)+tf.constant(1e-15))
    #     recall=diag_part/(tf.reduce_sum(cm,1)+tf.constant(1e-15))
    #     f1=2*precision*recall/(precision+recall+tf.constant(1e-15))
    #     return precision,recall,f1

def gen_confusion_matrix(y_true, y_pred):
    
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(tf.round(y_pred), tf.int32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred, num_classes=2)
    if (tf.size(y_true) >= 32):
        tf.print(cm)
    return cm
    
class class_accuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super(class_accuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name="acc", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # tf.print(y_true)
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

