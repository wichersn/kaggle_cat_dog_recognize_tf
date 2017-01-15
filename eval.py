import tensorflow as tf
import input
import model
import time

filenames, labels = input.get_filenames_labels(12500, .90, False, "../train_preprocessed2")

print("num_examples", len(filenames))

batch_size = 10
x, y_ = input.input_pipeline(filenames, labels, batch_size, isTrain=False)

sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

with tf.variable_scope("model") as scope:
    y = model.model(x, False)

error = model.get_error(y, y_)

merged_summary_op = model.get_summary_op(x, None, error)

logs_path = "../eval_logs"
model_dir = "../logs/"
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

sess.run(tf.global_variables_initializer())

run_num = 0
while True:
    print("start calc")
    saver = tf.train.Saver()
    try:
        checkpoint_state = tf.train.get_checkpoint_state(model_dir)
        saver.restore(sess, checkpoint_state.model_checkpoint_path)
    except:
        print("No model found")
    else:

        num_iters = int(len(filenames) / batch_size)

        i = 0
        total_error = 0
        while not coord.should_stop() and i < num_iters:
            error_val = sess.run(error)
            total_error += error_val
            i += 1


        average_error = total_error/float(num_iters)

        summary = tf.Summary()
        summary.ParseFromString(sess.run(merged_summary_op))
        summary.value.add(tag='Average Error', simple_value=average_error)
        summary_writer.add_summary(summary, run_num)

        print("average error", average_error)

    run_num += 1

    time.sleep(600)

coord.request_stop()

coord.join(threads)

sess.close()