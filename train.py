import tensorflow as tf
import input
import model

filenames, labels = input.get_filenames_labels(30, .8, True, "../train_processed")

x, y_ = input.input_pipeline(filenames, labels, 20)


sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

logs_path = "../logs"
summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

y = model.model(x)
loss = model.get_loss(y, y_)
optimizer =model.get_optimizer(loss)

merged_summary_op = model.get_summary_op(x)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

i = 0
try:
    while not coord.should_stop() and i < 10:
        _, summary = sess.run([optimizer, merged_summary_op])
        summary_writer.add_summary(summary, i)
        # print(x.eval(session=sess))
        i += 1

        if i % 5 == 0:
            save_path = saver.save(sess, "../saved_models/model.ckpt")

except tf.errors.OutOfRangeError:
    print("Done")
finally:
    coord.request_stop()

coord.join(threads)

sess.close()