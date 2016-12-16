import tensorflow as tf
import input
import model
import time

#tf.set_random_seed(68406)

filenames, labels = input.get_filenames_labels(12500, .95, True, "../train")

images, batch_labels = input.input_pipeline(filenames, labels, 50)


x = tf.placeholder(tf.float32, shape=images.get_shape())
y_ = tf.placeholder(tf.float32, shape=batch_labels.get_shape())

sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

mutable_x = tf.Variable(images)
assign_op = mutable_x.assign(x)

with tf.variable_scope("model") as scope:
    y = model.model(x, True)

    scope.reuse_variables()
    adver_y = model.model(mutable_x, False)

loss = model.get_loss(y, y_)
error = model.get_error(y, y_)
optimizer =model.get_optimizer(loss)

shifted_y_ = tf.concat(1, [tf.slice(y_, [0, 1], [-1, 1]), tf.slice(y_, [0, 0], [-1, 1])])
adver_loss = model.get_loss(adver_y, shifted_y_)
adver_optimizer = tf.train.GradientDescentOptimizer(learning_rate=10).minimize(
    adver_loss, var_list=[mutable_x])

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
last_save_time = time.time()
# print(assign_op)
# print(x)
# print(mutable_x)
# print(adver_optimizer)
try:
    while not coord.should_stop() and i < 100000:
        images_val, batch_labels_val = sess.run([images, batch_labels])
        _, x_val, mutable_x_val = sess.run([assign_op, x, mutable_x], feed_dict={x: images_val, y_: batch_labels_val})
        x_val, mutable_x_val, _ = sess.run([x, mutable_x, adver_optimizer], feed_dict={x: images_val, y_: batch_labels_val})
        mutable_x_val = mutable_x.eval(session=sess)
        sess.run(optimizer, feed_dict={x: mutable_x_val.tolist(), y_: batch_labels_val})
        # print(x.eval(session=sess))

        if time.time() >= last_summary_time + 30:
        #if i % 250 == 0:
            #summary = sess.run(merged_summary_op)

            #summary_writer.add_summary(summary, i)
            last_summary_time = time.time()
            print("summary", i)

        if time.time() >= last_save_time + 20:
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