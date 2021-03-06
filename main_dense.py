import tensorflow as tf
import numpy as np
import os
import random
import imageio
from pathlib import Path
import ops
''' Part of code from https://github.com/taki0112/Densenet-Tensorflow'''

# Hyperparameter
f_dim = 64
nb_block = 5  # number of dense block
init_learning_rate = 0.5 * 1e-4
epsilon = 1e-8  # epsilon for AdamOptimizer
batch_size = 32

total_epochs = 100
input_size = 256
dropout_rate = 0.2

# read img id and labels
img_lst = list(Path('data/streets').glob('*/*.jpeg'))
labels = [p.parts[-2] for p in img_lst]
label_index = {l: i for i, l in enumerate(set(labels))}
print('label_index:  ', label_index)
class_num = len(label_index)

train_ratio = 0.95
train_imgslst = img_lst[0:int(len(img_lst) * train_ratio)]
test_imgslst = img_lst[int(len(img_lst) * train_ratio):]

x = tf.compat.v1.placeholder(tf.float32,
                             shape=[None, input_size, input_size, 3])
batch_images = tf.reshape(x, [-1, input_size, input_size, 3])
label = tf.compat.v1.placeholder(tf.float32, shape=[None, class_num])

learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

logits = ops.DenseNet(x=batch_images,
                      nb_blocks=nb_block,
                      filters=f_dim,
                      class_num=class_num,
                      dropout_rate=dropout_rate).model
cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.stop_gradient(label), logits=logits))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate,
                                             epsilon=epsilon)
train = optimizer.minimize(cost)

correct_prediction = tf.equal(tf.argmax(input=logits, axis=1),
                              tf.argmax(input=label, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

tf.compat.v1.summary.scalar('loss', cost)
tf.compat.v1.summary.scalar('accuracy', accuracy)

saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

model_name = 'NET.model'

# create/restore checkpoint directory
continue_from = None
continue_from_iteration = None
checkpoint_dir = 'checkpoint_dense_train_new'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
runs = sorted(map(int, next(os.walk(checkpoint_dir))[1]))
if len(runs) == 0:
    run_nr = 0
else:
    run_nr = runs[-1] + 1
run_folder = str(run_nr).zfill(3)

checkpoint_dir = os.path.join(checkpoint_dir, run_folder)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
save_dir = checkpoint_dir

# build up graph
with tf.compat.v1.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and tf.compat.v1.train.checkpoint_exists(
            ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.compat.v1.global_variables_initializer())

    if continue_from:
        checkpoint_dir = os.path.join(os.path.dirname(checkpoint_dir),
                                      continue_from)
        print('Loading variables from ' + checkpoint_dir)
        ops.load_checkpoint(sess, checkpoint_dir, continue_from_iteration,
                            model_name)
    if continue_from_iteration:
        epoch_start = continue_from_iteration + 1
    else:
        epoch_start = 0

    merged = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter('./logs', sess.graph)

    global_step = 0

    for epoch in range(epoch_start, total_epochs):

        epoch_learning_rate = init_learning_rate * (0.95**(epoch // 2))

        total_batch = int(len(train_imgslst) / batch_size)
        idx = random.sample(range(0, len(train_imgslst)), len(train_imgslst))

        for step in range(total_batch):
            batch_idx = idx[step * batch_size:(step + 1) * (batch_size)]
            batch_x = []
            batch_y = []

            for ii in range(batch_size):
                batch_img = imageio.imread(train_imgslst[batch_idx[ii]])
                batch_x.append(batch_img)

                label_vector = np.zeros(class_num)
                img_class = train_imgslst[batch_idx[ii]].stem.split('_')[1]
                label_vector[int(label_index[img_class])] = 1
                batch_y.append(label_vector)

            batch_img = np.array(batch_x)
            batch_label = np.asarray(batch_y)

            train_feed_dict = {
                x: batch_img,
                label: batch_label,
                learning_rate: epoch_learning_rate
            }

            _, loss, train_accuracy = sess.run([train, cost, accuracy],
                                               feed_dict=train_feed_dict)
            print("Step:", str(global_step), "Loss:", loss,
                  "Training accuracy:", train_accuracy)

            global_step += 1

        saver.save(sess,
                   save_path=os.path.join(save_dir, model_name),
                   global_step=epoch)

        # randomly choose batch_size amount of images for validation
        test_x = []
        test_y = []
        ti_idx = random.sample(range(0, len(test_imgslst)), batch_size)
        for ti in range(batch_size):
            test_x_tmp = imageio.imread(test_imgslst[ti_idx[ti]])
            test_x.append(test_x_tmp)
            label_vector = np.zeros(class_num)
            img_class = test_imgslst[ti_idx[ti]].parts[-2]
            label_vector[int(label_index[img_class])] = 1
            test_y.append(label_vector)

        test_feed_dict = {
            x: test_x,
            label: test_y,
            learning_rate: epoch_learning_rate
        }

        accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
        print('Epoch:', '%04d' % (epoch), '/ Accuracy =', accuracy_rates)

        # compute test accuracy after 3 epochs
        if epoch > 3:
            accuracy_all = 0
            for ti in range(len(test_imgslst)):
                test_x = []
                test_y = []
                test_x_tmp = imageio.imread(test_imgslst[ti])
                test_x.append(test_x_tmp)
                label_vector = np.zeros(class_num)
                img_class = test_imgslst[ti].parts[-2]
                label_vector[int(label_index[img_class])] = 1
                test_y.append(label_vector)

                test_feed_dict = {
                    x: test_x,
                    label: test_y,
                    learning_rate: epoch_learning_rate
                }
                accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
                accuracy_all = accuracy_all + accuracy_rates

            print('Epoch total:', '%04d' % (epoch), '/ Accuracy =',
                  accuracy_all / int(len(test_imgslst)))
