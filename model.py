import tensorflow as tf

def model(x):

    conv1 = tf.contrib.layers.convolution2d(x, 10, [10,10], [4,4], "VALID",
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            activation_fn=tf.nn.relu)
    conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    print(conv1)

    # last_conv = tf.contrib.layers.convolution2d(conv1, 5, [4,4], [2,2], "VALID", activation_fn=tf.nn.relu)
    # last_conv = tf.nn.max_pool(last_conv, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    # print(last_conv)

    shape = conv1.get_shape().as_list()
    reshaped_last_conv = tf.reshape(conv1, [-1, shape[1] * shape[2] * shape[3]])

    w = tf.Variable(tf.zeros((shape[1] * shape[2] * shape[3], 2), dtype=tf.float32))
    b = tf.Variable(tf.zeros(2, dtype=tf.float32))
    y = tf.matmul(reshaped_last_conv, w) + b

    #y = tf.contrib.layers.fully_connected(reshaped_last_conv, 2,
    #                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.1))

    return y

def get_loss(y, y_):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

def get_optimizer(loss):
    return tf.train.AdamOptimizer().minimize(loss)

def get_error(y, y_):
    pred = tf.argmax(y, 1)
    return 1 - tf.reduce_mean(tf.to_float(tf.equal(pred, tf.argmax(y_, 1))))

def get_summary_op(x, loss):
    tf.image_summary("images", x, max_images=3)
    tf.summary.scalar("loss", loss)
    return tf.summary.merge_all()