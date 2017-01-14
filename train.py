import tensorflow as tf
import input
import model

with tf.variable_scope("input"):
    filenames, labels = input.get_filenames_labels(12500, .95, True, "../train_preprocessed2")
    x, y_ = input.input_pipeline(filenames, labels, 70)

with tf.variable_scope("model") as scope:
    y = model.model(x, True)

with tf.variable_scope("optimizer"):
    loss = model.get_loss(y, y_)
    optimizer =model.get_optimizer(loss)

with tf.variable_scope("error"):
    error = model.get_error(y, y_)

saver = tf.train.Saver()

with tf.variable_scope("summary"):
    logs_path = "../logs"
    merged_summary_op = model.get_summary_op(x, loss, error)

sv = tf.train.Supervisor(logdir="../logs",
                         init_op=tf.global_variables_initializer(),
                         summary_op=merged_summary_op,
                         saver=saver,
                         save_summaries_secs=30,
                         save_model_secs=60)

with sv.managed_session() as sess:
    while not sv.should_stop():
        sess.run(optimizer)