import tensorflow as tf


def center_loss(features, label, nrof_classes=8, alfa=0.95):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[-1]
    features = tf.reshape(features, [-1, nrof_features.value])
    centers = tf.get_variable(
        'centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False
    )
    label = tf.cast(tf.argmax(label, 1), tf.int32)
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def focal_loss(pred, label, softmax_loss, garma=2, alpha=0.25):
    pred_t = label * pred + (1 - label) * (1 - pred)
    alpha_t = label * alpha + (1 - label) * (1 - alpha)
    focal_loss = alpha_t * tf.pow((1 - pred_t), garma) * softmax_loss
    return tf.reduce_sum(focal_loss)


def inverse_focal_loss(pred, label, softmax_loss, garma=2, alpha=0.25):
    pred_t = label * (1 - pred) + (1 - label) * pred
    alpha_t = label * (1 - alpha) + (1 - label) * alpha
    focal_loss = alpha_t * tf.pow((1 - pred_t), garma) * softmax_loss
    return tf.reduce_sum(focal_loss)


def arcface_loss(W, X, label, m=0.1, s=3):
    X = tf.squeeze(X)
    WN = tf.norm(W, axis=0, keep_dims=True)
    XN = tf.norm(X, axis=1, keep_dims=True)
    cos_theta = tf.matmul(X / XN, W / WN)
    cos_theta_yi = tf.reduce_sum(label * cos_theta, axis=1, keep_dims=True)
    sigma_exp_cos_theta = tf.reduce_sum(tf.exp(s * cos_theta * (1 - label)), axis=1, keep_dims=True)
    # cos(θ_yi + m) = cos θ_yi cos m − sin θ_yi sin m
    sin_theta_yi = tf.norm(1 - cos_theta_yi ** 2, axis=1, keep_dims=True) ** 0.5
    exp_cos_theta_yi_m = tf.exp(s * (cos_theta_yi * tf.cos(m) - sin_theta_yi * tf.sin(m)))
    E = exp_cos_theta_yi_m / (exp_cos_theta_yi_m + sigma_exp_cos_theta)
    return -tf.reduce_mean(tf.log(E + 10e-5))


def MSMSE(layer_dict, label):
    loss = 0
    for i in range(4):
        loss += tf.losses.mean_squared_error(label[i], layer_dict['heat_map_' + str(i + 1)])
    return loss
