import tensorflow as tf
import input
import model
import time

sess = tf.Session()

with tf.variable_scope("input"):
    filenames, labels = input.get_filenames_labels(12500, .95, True, "../train")
    x, y_ = input.input_pipeline(filenames, labels, 70)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

with tf.variable_scope("model") as scope:
    y = model.model(x, True)

with tf.variable_scope("optimizer"):
    loss = model.get_loss(y, y_)
    optimizer =model.get_optimizer(loss)

with tf.variable_scope("error"):
    error = model.get_error(y, y_)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
model_path = "../logs/model.ckpt"
try:
    saver.restore(sess, model_path)
except tf.errors.NotFoundError:
    print("No previous model")

with tf.variable_scope("summary"):
    logs_path = "../logs"
    merged_summary_op = model.get_summary_op(x, loss, error)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

i = 0
last_summary_time = 0
last_save_time = 0 #time.time()
try:
    while not coord.should_stop() and i < 100000:
        sess.run(optimizer)
        # print(x.eval(session=sess))

        if time.time() >= last_summary_time + 30:
        #if i % 250 == 0:
            summary = sess.run(merged_summary_op)

            summary_writer.add_summary(summary, i)
            last_summary_time = time.time()
            print("summary", i)

        if time.time() >= last_save_time + 30:
        #if i % 250 == 0:
            save_path = saver.save(sess, model_path)
            last_save_time = time.time()
            print("saved", i)

        i += 1

except tf.errors.OutOfRangeError:
    print("Done")
finally:
    coord.request_stop()

coord.join(threads)

sess.close()