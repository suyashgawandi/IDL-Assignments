import tensorflow as tf
import numpy as np


# get the data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


def preprocess_images(images):
    return images.reshape(-1, 784).astype(np.float32) / 255


def preprocess_labels(labels):
    return labels.reshape(-1).astype(np.int32)


train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
train_labels = preprocess_labels(train_labels)
test_labels = preprocess_labels(test_labels)

train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(128).repeat()
#test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(128)


# define the model first, from input to output

# uhm, maybe don't use that many layers actually. 2 is fine!
n_units = 100
n_layers = 2
w_range = 0.1

# just set up a "chain" of hidden layers
# model is represented by a list where each element is a layer,
# and each layer is in turn a list of the layer variables (w, b)

# first layer goes from n_input to n_hidden
w_input = tf.Variable(tf.random.uniform([784, n_units], -w_range, 0.),
                      name="w0")
b_input = tf.Variable(tf.zeros(n_units), name="b0")
layers = [[w_input, b_input]]

# all other hidden layers go from n_hidden to n_hidden
for layer in range(n_layers - 1):
    w = tf.Variable(tf.random.uniform([n_units, n_units], -w_range, 0.),
                    name="w" + str(layer+1))
    b = tf.Variable(tf.zeros(n_units), name="b" + str(layer+1))
    layers.append([w, b])

# finally add the output layer
w_out = tf.Variable(tf.random.uniform([n_units, 10], -w_range, 0.),
                    name="wout")
b_out = tf.Variable(tf.zeros(10), name="bout")
layers.append([w_out, b_out])

# flatten the layers to get a list of variables
all_variables = [variable for layer in layers for variable in layer]


def model_forward(inputs):
    x = inputs
    for w, b in layers[:-1]:
        x = tf.nn.relu(tf.matmul(x, w) + b)
    logits = tf.matmul(x, layers[-1][0]) + layers[-1][1]

    return logits


lr = 0.1
train_steps = 2000
for step, (img_batch, lbl_batch) in enumerate(train_data):
    if step > train_steps:
        break

    with tf.GradientTape() as tape:
        # here we just run all the layers in sequence via a for-loop
        logits = model_forward(img_batch)
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=lbl_batch))

    grads = tape.gradient(xent, all_variables)
    for grad, var in zip(grads, all_variables):
        var.assign_sub(lr*grad)

    if not step % 100:
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch), tf.float32))
        print("Loss: {} Accuracy: {}".format(xent, acc))


test_preds = model_forward(test_images)
test_preds = tf.argmax(test_preds, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, test_labels), tf.float32))
print("Final test accuracy: {}".format(acc))
