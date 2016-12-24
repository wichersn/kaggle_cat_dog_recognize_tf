import tensorflow as tf
import model
import input
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

            with tf.variable_scope("input"):
                filenames, labels = input.get_filenames_labels(12500, .95, True, "../train")
                x, y_ = input.input_pipeline(filenames, labels, 70)

                coord = tf.train.Coordinator()

            with tf.variable_scope("model"):
                y = model.model(x, True)

            with tf.variable_scope("optimizer"):
                global_step = tf.Variable(0)
                loss = model.get_loss(y, y_)
                train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))

            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            non_model_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) - set(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))

        print("model setup")

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="../logs",
                                 init_op=init_op,
                                 local_init_op =tf.variables_initializer(non_model_vars),
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=30)

        print("superviser created")

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            print("sess created")

            # Loop until the supervisor shuts down or 1000000 steps have completed.
            tf.train.start_queue_runners(coord=coord, sess=sess)
            step = 0
            start_time = time.time()
            while not sv.should_stop() and step < 1000000:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                _, step = sess.run([train_op, global_step])

        print("exit")
        # Ask for all the services to stop.
        sv.stop()

if __name__ == "__main__":
    tf.app.run()