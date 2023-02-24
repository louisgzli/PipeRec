import os
import time
import tensorflow as tf


def get_log_str(logs, prefix_str=""):
    str_buffer = prefix_str
    for key in sorted(logs.keys()):
        value = logs.get(key, 0)
        str_buffer = str_buffer + f", {key} ({value:.2f})"
    return str_buffer


class NBatchLogger(tf.keras.callbacks.Callback):

    def __init__(self, batch_size, n_batch=10):
        """
        每N个batch记录信息的回调函数
        :param batch_size: batch size
        :param n_batch: 每N个batch
        """
        self.batch_size = batch_size
        self.n_batch = n_batch
        self.start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.n_batch == 0:
            qps = self.n_batch * self.batch_size / (time.time() - self.start_time)
            str_buffer = f"batch ({batch}), speed ({qps:.2f} qps/s)"
            log_str = get_log_str(logs if logs else {}, str_buffer)
            self.start_time = time.time()
            print(log_str)

    def on_predict_batch_end(self, batch, logs=None):
        if (batch + 1) % self.n_batch == 0:
            qps = self.n_batch * self.batch_size / (time.time() - self.start_time)
            self.start_time = time.time()
            print("batch ({}), speed ({:.2f} qps/s)".format(batch, qps))


class ModelCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, model_save_root, model_name):
        self.model_save_root = model_save_root
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        i = epoch + 1
        save_path = os.path.join(self.model_save_root, f"epoch{i}", self.model_name)
        self.model.save_weights(save_path)
