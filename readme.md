Kaggle competition: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition You can download the data there.

This gets around 87% accuracy on the problem.

Run preprocess_images.py first before training to resize them. Then use train.py to train the network. This periodically saves the model and you can evaluate the saved model with eval.py in parallel to training. 

You can also train the model on multiple machines with train_distributed.py.
Sample options to use with train_distrubuted:

--ps_hosts=192.168.0.16:2221 --worker_hosts=192.168.0.16:2222,192.168.0.15:2222 --job_name=ps --task_index=0

You’ll also have to run a similar command on another computer with a different task_index. For more information on distributed training see https://www.tensorflow.org/deploy/distributed

Adversarial_train_3.py modifies the training examples to fool the network by making the cats slightly more dog like, and the dogs slightly more cat like. This should help the network do better on test data since the network will learn that small changes to the image should still produce the same result, and it will reveal more of the networks flaws during training. See https://arxiv.org/pdf/1412.6572.pdf for more about adversarial examples. For some reason, I haven’t seen any improvement in test accuracy from adding adversarial examples.
