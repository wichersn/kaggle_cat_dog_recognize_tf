import tensorflow as tf

def get_total_size(t):
    shape = t.get_shape().as_list()
    return shape[1] * shape[2] * shape[3]

def model(x, isTrain, train_keep_prob=.75):
    print(get_total_size(x))

    with tf.variable_scope("inception1"):
        conv1 = inseption_module(x, isTrain, five_conv_size=6, three_conv_size=3, ave_pool_size=4, one_one_ave_size=2, max_pool_size=4, train_keep_prob=train_keep_prob)
    print(conv1)
    print(get_total_size(conv1))

    with tf.variable_scope("inception2"):
        conv2 = inseption_module(conv1, isTrain, five_conv_size=15, three_conv_size=10, ave_pool_size=3, one_one_ave_size=7, max_pool_size=3, train_keep_prob=train_keep_prob)
    print(conv2)
    print(get_total_size(conv2))

    with tf.variable_scope("inception3"):
        last_conv = inseption_module(conv2, isTrain, five_conv_size=25, three_conv_size=17, ave_pool_size=2, one_one_ave_size=15, max_pool_size=3, train_keep_prob=train_keep_prob)
    print(last_conv)
    print(get_total_size(last_conv))

    shape = last_conv.get_shape().as_list()
    reshaped_last_conv = tf.reshape(last_conv, [-1, shape[1] * shape[2] * shape[3]])

    print(reshaped_last_conv)

    with tf.variable_scope("fully_connect"):
        fully_connect = tf.contrib.layers.fully_connected(reshaped_last_conv, 500,
                                                          biases_initializer=tf.constant_initializer(0.0),
                                                          weights_initializer=tf.contrib.layers.xavier_initializer()
                                                          )
        if isTrain:
            fully_connect = tf.nn.dropout(fully_connect, train_keep_prob)

    with tf.variable_scope("output"):
        y = tf.contrib.layers.fully_connected(fully_connect, 2,
                                              biases_initializer=tf.constant_initializer(0.0),
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation_fn=None
                                              )

    return y

def inseption_module(x, isTrain, five_conv_size, three_conv_size, ave_pool_size, one_one_ave_size, max_pool_size, train_keep_prob=.5):
    one_one_conv = tf.contrib.layers.convolution2d(x, x.get_shape().as_list()[3], [1, 1], [1, 1], "SAME",
                                                  weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                  biases_initializer=tf.constant_initializer(0.0),
                                                  activation_fn=tf.nn.relu, scope="1x1"
                                                  )
    five_conv = tf.contrib.layers.convolution2d(one_one_conv, five_conv_size, [5, 5], [1, 1], "SAME",
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               biases_initializer=tf.constant_initializer(0.0),
                                               activation_fn=tf.nn.relu, scope="5x5"
                                               )
    three_conv = tf.contrib.layers.convolution2d(one_one_conv, three_conv_size, [3, 3], [1, 1], "SAME",
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.0),
                                                activation_fn=tf.nn.relu, scope="3x3"
                                                )
    ave_pool = tf.nn.avg_pool(x, [1, ave_pool_size, ave_pool_size, 1], [1, 1, 1, 1], "SAME")
    one_one_ave_conv = tf.contrib.layers.convolution2d(ave_pool, one_one_ave_size, [1, 1], [1, 1], "SAME",
                                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                       biases_initializer=tf.constant_initializer(0.0),
                                                       activation_fn=tf.nn.relu, scope="ave_X1"
                                                       )
    combined_conv = tf.concat(3, (five_conv, three_conv, one_one_ave_conv))

    result = tf.nn.max_pool(combined_conv, [1, max_pool_size, max_pool_size, 1], [1, 2, 2, 1], 'VALID')
    if isTrain:
        result = tf.nn.dropout(result, train_keep_prob)
    return tf.nn.lrn(result, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def get_loss(y, y_):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

def get_optimizer(loss):
    return tf.train.AdamOptimizer().minimize(loss)

def get_error(y, y_):
    pred = tf.argmax(y, 1)
    return 1 - tf.reduce_mean(tf.to_float(tf.equal(pred, tf.argmax(y_, 1))))

def get_summary_op(x, loss, error):
    tf.summary.image("images", x, max_outputs=10)
    if not loss == None:
        tf.summary.scalar("loss", loss)
    if not error == None:
        tf.summary.scalar("error", error)
    return tf.summary.merge_all()