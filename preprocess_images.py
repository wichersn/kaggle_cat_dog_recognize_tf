import tensorflow as tf
import input
import matplotlib.image as mpimg

filenames, _ = input.get_filenames_labels(12500, 1.0, True, "../train")

example = tf.placeholder(tf.uint8, shape=[None, None, 3])
print(example)

resized_example = tf.image.resize_images(example, [50, 50])

resized_example = tf.cast(resized_example*255/tf.reduce_max(resized_example), tf.uint8)

encoded_example = tf.image.encode_jpeg(resized_example)

#resized_example = tf.image.encode_jpeg(example)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for filename in filenames:
    img = mpimg.imread(filename)

    new_filename = filename.replace("../train", "../train_preprocessed2")
    file = open(new_filename, "wb+")
    file.write(encoded_example.eval(feed_dict = {example: img}, session=sess))
    file.close()