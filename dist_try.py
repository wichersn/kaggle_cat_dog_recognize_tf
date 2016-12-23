import tensorflow as tf
import time

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
        print("joined")
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            batch_size = 100000
            x = tf.placeholder(tf.float32, shape=(batch_size, 1))
            y_ = tf.placeholder(tf.float32, shape=(batch_size, 1))
            w = tf.Variable(tf.zeros([1, 1], dtype=tf.float32))
            loss = tf.reduce_mean(y_ - tf.matmul(x, w))

            global_step = tf.Variable(0)
            train_op = tf.train.GradientDescentOptimizer(.00001).minimize(loss, global_step=global_step)

        feed_val = [[1.0] for i in range(batch_size)]
        feed_dict = {x: feed_val, y_: feed_val}
        print(feed_dict)

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="../logs",
                                 init_op=tf.global_variables_initializer(),
                                 init_feed_dict=feed_dict,
                                 global_step=global_step,
                                 summary_op = None,
                                 save_model_secs=60)

        print("superviser created")

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        last_print_time = 0
        with sv.managed_session(server.target) as sess:
            print("sess created")

            # Loop until the supervisor shuts down or 1000000 steps have completed.
            step = 0
            while not sv.should_stop() and step < 1000000:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                _, step = sess.run([train_op, global_step], feed_dict=feed_dict)

                if time.time() >= last_print_time + 10:
                    print(sess.run(loss, feed_dict=feed_dict), step)
                    last_print_time = time.time()
                #time.sleep(.01)

        print("exit")
        # Ask for all the services to stop.
        sv.stop()

if __name__ == "__main__":
    tf.app.run()