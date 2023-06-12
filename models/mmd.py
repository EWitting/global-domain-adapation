# code from:
# https://www.idiap.ch/software/bob/docs/bob/bob.learn.tensorflow/v1.2.0/_modules/bob/learn/tensorflow/loss/mmd.html
# slightly modified to include 'beta'  parameter
"""
Maximum Mean Discrepancy (MMD)

The MMD is implemented as keras regularizer that can be used for
shared layers. This implementation uis tested under keras 1.1.0.

- Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
Advances in neural information processing systems. 2007.
"""
import tensorflow as tf


def compute_kernel(x, y, beta):
    """Gaussian kernel.
    """
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(
        tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1])
    )
    tiled_y = tf.tile(
        tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1])
    )
    return tf.exp(
        -beta * tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32)
    )


def mmd(x, y, beta):
    """Maximum Mean Discrepancy with Gaussian kernel.
    See: https://stats.stackexchange.com/a/276618/49433
    """
    x_kernel = compute_kernel(x, x, beta)
    y_kernel = compute_kernel(y, y, beta)
    xy_kernel = compute_kernel(x, y, beta)
    return (
        tf.reduce_mean(x_kernel)
        + tf.reduce_mean(y_kernel)
        - 2 * tf.reduce_mean(xy_kernel)
    )
