import tensorflow as tf

def get_filenames_labels(num_imgs, train_percent, is_train, dir):
    image_list = []
    label_list = []

    for i in range(num_imgs):
        image_list.append(dir + "/cat.{}.jpg".format(i))
        label_list.append([1, 0])

        image_list.append(dir + "/dog.{}.jpg".format(i))
        label_list.append([0, 1])

    train_num = int(len(image_list) * train_percent)

    if is_train:
        return image_list[:train_num], label_list[:train_num]
    else:
        return image_list[train_num:], label_list[train_num:]

def read_imgs_opp(input_queue):
    label = input_queue[1]
    image_file = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_file, channels=3)
    return image, label

def input_pipeline(filenames, labels, batch_size, isTrain=True):
    filename_queue = tf.train.slice_input_producer(
      [filenames, labels], shuffle=isTrain)
    example, label = read_imgs_opp(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size


    example = transform_example(example, isTrain)

    min_after_dequeue = 100
    capacity = min_after_dequeue + 2 * batch_size
    if isTrain:
        example_batch, label_batch = tf.train.shuffle_batch(
          [example, label], batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)
    else:
        example_batch, label_batch = tf.train.batch(
            [example, label], batch_size=batch_size, capacity=capacity)
        example_batch = tf.cast(example_batch, tf.float32)
    return example_batch, label_batch

def transform_example(example, isTrain):
    #example = tf.image.resize_images(example, [50, 50])  # tf.reshape(example, [100, 100, 3])
    example = tf.reshape(example, [50, 50, 3])
    if isTrain:
        example = tf.image.random_flip_left_right(example)
        example = tf.image.random_brightness(example, .7)
        example = tf.image.random_contrast(example, .4, 1.6)
    example = tf.image.per_image_standardization(example)
    if isTrain:
        noise = tf.truncated_normal(example.get_shape(), stddev=.1)
        example = example + noise


    return example