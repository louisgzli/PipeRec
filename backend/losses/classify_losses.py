import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy, CategoricalHinge


@tf.function
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测阶段则输出y_pred大于0的类。
         如有疑问，请仔细阅读并理解本文：https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


@tf.function
def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359
    """
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred = K.concatenate([y_pred, zeros], axis=-1)
    inf_vecs = zeros + 1e12

    if mask_zero:
        y_pred = K.concatenate([inf_vecs, y_pred[..., 1:]], axis=-1)

    y_pos_2 = tf.gather(y_pred, y_true, batch_dims=K.ndim(y_true)-1)
    y_pos_1 = K.concatenate([y_pos_2, zeros], axis=-1)
    if mask_zero:
        y_pred = K.concatenate([-inf_vecs, y_pred[..., 1:]], axis=-1)
        y_pos_2 = tf.gather(y_pred, y_true, batch_dims=K.ndim(y_true)-1)
    pos_loss = K.logsumexp(-y_pos_1, axis=-1)
    all_loss = K.logsumexp(y_pred, axis=-1)
    aux_loss = K.logsumexp(y_pos_2, axis=-1) - all_loss
    aux_loss = K.clip(1 - K.exp(aux_loss), K.epsilon(), 1)
    neg_loss = all_loss + K.log(aux_loss)
    return pos_loss + neg_loss


@tf.function
def sparse_categorical_crossentropy(y_true, y_pred):
    return SparseCategoricalCrossentropy(y_true, y_pred)


@tf.function
def binary_crossentropy(y_true, y_pred):
    return BinaryCrossentropy(y_true, y_pred)


@tf.function
def category_crossentropy(y_true, y_pred):
    return CategoricalCrossentropy(y_true, y_pred)


@tf.function
def category_hinge(y_true, y_pred):
    return CategoricalHinge(y_true, y_pred)
