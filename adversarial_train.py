import tensorflow as tf
import input
import model
import time

#tf.set_random_seed(68406)

sess = tf.Session()

with tf.variable_scope("input") as scope:
    filenames, labels = input.get_filenames_labels(12500, .95, True, "../train")
    images, batch_labels = input.input_pipeline(filenames, labels, 1)

    x = tf.Variable(images, trainable=False)
    x_assign_opp = x.assign(images)

    y_ = tf.Variable(batch_labels, trainable=False)
    y__assign_opp = y_.assign(batch_labels)
    #x = x.assign(images)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


with tf.variable_scope("model") as scope:
    #adver_y = model.model(x, False)

    #scope.reuse_variables()
    #with tf.get_default_graph().control_dependencies([x_assign_opp]):
        y = model.model(x, True)

loss = model.get_loss(y, y_)
error = model.get_error(y, y_)

print(x)

model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
print(len(model_vars))
print(len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
optimizer = tf.train.AdamOptimizer().minimize(loss, var_list=model_vars[:])

# shifted_y_ = tf.concat(1, [tf.slice(y_, [0,1], [-1,1]), tf.slice(y_, [0,0], [-1,1])])
# adver_loss = model.get_loss(adver_y, shifted_y_)
# adver_optimizer = tf.train.GradientDescentOptimizer(learning_rate=.00000001).minimize(
#     adver_loss, var_list=[x])

assign_both = tf.group(x_assign_opp, y__assign_opp)

merged_summary_op = model.get_summary_op(x, loss, error)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list=model_vars)
try:
    saver.restore(sess, "../saved_models/model.ckpt")
except tf.errors.NotFoundError:
    print("No previous model")

logs_path = "../logs"
summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

#with sess.as_default():
    #print(x.eval())
    #sess.run([x_assign_opp])
    #print(x.eval())
    #print(images.eval() == images.eval())

i = 0
last_summary_time = 0
last_save_time = time.time()
try:
    while not coord.should_stop() and i < 100000:
        with sess.as_default():
            #sess.run(adver_optimizer)
            #sess.run([x_assign_opp])
            sess.run([assign_both, optimizer])
            #sess.run([optimizer])
            #sess.run(assign_optimize)
            # print(x.eval(session=sess))

        if time.time() >= last_summary_time + 10:
        #if i % 250 == 0:
            summary = sess.run(merged_summary_op)

            summary_writer.add_summary(summary, i)
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