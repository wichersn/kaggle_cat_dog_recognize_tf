import tensorflow as tf
import input
import model
import time

filenames, labels = input.get_filenames_labels(12500, .90, True, "../train_preprocessed2")

batch_size = 70
images, y_ = input.input_pipeline(filenames, labels, batch_size)


sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


with tf.variable_scope("model") as scope:
    adver_y = model.model(images, False)

shifted_y_ = tf.concat(1, [tf.slice(y_, [0,1], [-1,1]), tf.slice(y_, [0,0], [-1,1])])
adver_loss = model.get_loss(adver_y, shifted_y_)

grad = tf.gradients(adver_loss, images)[0]

#scale_grad = tf.abs(tf.truncated_normal(shape=grad.get_shape(), stddev=.01))
update_prob = .1
update_mag = .01
scale_grad = tf.to_float(tf.random_uniform(shape=[batch_size]) > update_prob) * update_mag
grad_shape = grad.get_shape().as_list()
scale_grad = tf.tile(scale_grad, [grad_shape[1] * grad_shape[2] * grad_shape[3]])
scale_grad = tf.reshape(scale_grad, grad_shape[1:4] + [batch_size])
scale_grad = tf.transpose(scale_grad, [3, 0, 1, 2])
update = -tf.mul(tf.sign(grad), scale_grad)

new_x = images + update
#new_x = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), new_x)

x = tf.Variable(images, trainable=False)
x = x.assign(new_x)
print("x", x)
print("x", x)


with tf.variable_scope("model") as scope:
    scope.reuse_variables()
    y = model.model(x, True, .8)

loss = model.get_loss(y, y_)
error = model.get_error(y, y_)
optimizer =model.get_optimizer(loss)

tf.summary.image("images", images, max_outputs=20)
tf.summary.image("x", x, max_outputs=20)
tf.summary.image("update", update, max_outputs=20)
tf.summary.image("scale_grad", scale_grad, max_outputs=20)
tf.summary.scalar("loss", loss)
tf.summary.scalar("error", error)
merged_summary_op = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
model_dir = "../logs/"
try:
    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    saver.restore(sess, checkpoint_state.model_checkpoint_path)
except:
    print("No model found")

logs_path = "../logs"
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

i = 0
last_summary_time = 0
last_save_time = 0 #time.time()
try:
    while not coord.should_stop() and i < 100000:
        sess.run([optimizer, grad])
        # print(x.eval(session=sess))

        if time.time() >= last_summary_time + 60:
        #if i % 250 == 0:
            summary = sess.run(merged_summary_op)

            summary_writer.add_summary(summary, i)
            last_summary_time = time.time()
            print("summary", i)

        if time.time() >= last_save_time + 600:
        #if i % 250 == 0:
            save_path = saver.save(sess, model_dir + "/model.ckpt")
            last_save_time = time.time()
            print("saved", i)

        i += 1

except tf.errors.OutOfRangeError:
    print("Done")
finally:
    coord.request_stop()

coord.join(threads)

sess.close()