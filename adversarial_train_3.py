import tensorflow as tf
import input
import model
import time

filenames, labels = input.get_filenames_labels(12500, .95, True, "../train")

images, y_ = input.input_pipeline(filenames, labels, 50)


sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)




with tf.variable_scope("model") as scope:
    adver_y = model.model(images, False)

shifted_y_ = tf.concat(1, [tf.slice(y_, [0,1], [-1,1]), tf.slice(y_, [0,0], [-1,1])])
adver_loss = model.get_loss(adver_y, shifted_y_)

grad = tf.gradients(adver_loss, images)[0]

x = tf.Variable(images, trainable=False)
x = x.assign(images - grad*1000)


with tf.variable_scope("model") as scope:
    scope.reuse_variables()
    y = model.model(x, True)

loss = model.get_loss(y, y_)
error = model.get_error(y, y_)
optimizer =model.get_optimizer(loss)

tf.image_summary("images", images, max_images=10)
tf.image_summary("x", x, max_images=10)
tf.image_summary("grad", grad, max_images=10)
tf.summary.scalar("loss", loss)
tf.summary.scalar("error", error)
merged_summary_op = tf.summary.merge_all()

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
        sess.run([optimizer, grad])
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