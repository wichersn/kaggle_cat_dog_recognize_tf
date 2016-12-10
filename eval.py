import tensorflow as tf
import input
import model

filenames, labels = input.get_filenames_labels(12500, .8, False, "../train_processed")

x, y_ = input.input_pipeline(filenames, labels, 20)

sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

logs_path = "../logs"
summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

y = model.model(x)
error = model.get_error(y, y_)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

saver.restore(sess, "../saved_models/model.ckpt")

batch_size = 1
num_iters = 10 #len(filenames) / batch_size

i = 0
total_error = 0
try:
    while not coord.should_stop() and i < 10:
        error_val = sess.run(error)
        total_error += error_val
        # print(x.eval(session=sess))
        i += 1

except tf.errors.OutOfRangeError:
    print("Done")
finally:
    coord.request_stop()

coord.join(threads)

sess.close()

print("Average error", total_error/float(num_iters))