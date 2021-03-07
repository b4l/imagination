import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import densenet
''' Part of code from https://github.com/taki0112/Densenet-Tensorflow'''

# Hyperparameter
f_dim = 64
nb_block = 5  # number of dense block
initial_learning_rate = 0.5 * 1e-4
epsilon = 1e-8  # epsilon for AdamOptimizer
dropout_rate = 0.2
epochs = 10

# create dataset
batch_size = 32
img_height = 256
img_width = 256

data_dir = pathlib.Path('data/streets')

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=8484,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=8484,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
class_num = len(class_names)

# configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = densenet.DenseNet(nb_block,
                          class_num,
                          f_dim,
                          dropout_rate,
                          input_shape=(img_height, img_width, 3))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

# compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                       epsilon=epsilon),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
model.summary()

# save weights
cp_dir = pathlib.Path("checkpoints/" + model.name)
if cp_dir.exists():
    runs = [p for p in cp_dir.iterdir() if p.is_dir]
    last = sorted(runs)[-1].parts[-1] if len(runs) > 0 else 0
    cp_dir = cp_dir.joinpath(str(int(last) + 1).zfill(3))
else:
    cp_dir = cp_dir.joinpath(str(int(0) + 1).zfill(3))
cp_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = str(cp_dir) + "/cp-{epoch:04d}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# train model
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    callbacks=[cp_callback])

# visualize
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
