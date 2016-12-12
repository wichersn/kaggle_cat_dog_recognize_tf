import tensorflow as tf
import input
import model
import time

#tf.set_random_seed(68406)

filenames, labels = input.get_filenames_labels(12500, .95, True, "../train")

x, y_ = input.input_pipeline(filenames, labels, 50)


sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

y = model.model(x, True)
loss = model.get_loss(y, y_)
error = model.get_error(y, y_)
optimizer =model.get_optimizer(loss)

merged_summary_op = model.get_summary_op(x, loss, error)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
try:
    saver.restore(sess, "../saved_models/model.ckpt")
except tf.errors.NotFoundError:
    print("No previous model")

logs_path = "../logs"
summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

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

        if time.time() >= last_save_time + 60:
        #if i % 250 == 0:
            save_path = saver.save(sess, "../saved_models/model.ckpt")
            last_save_time = time.time()
            print("saved", i)

        i += 1

except tf.errors.OutOfRangeError:
    print("Done")
finally:
    coord.request_stop()

coord.join(threads)

sess.close()