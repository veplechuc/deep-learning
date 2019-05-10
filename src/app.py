import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


sess = tf.InteractiveSession()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def display_sample(num):
    # display the selected num from the mnist dataset

    # the one_hot array
    print(mnist.train.labels[num])
    label = mnist.train.labels[num].argmax(axis=0)
    # reshape the 784 linear representation into a 28x28 image
    image = mnist.train.images[num].reshape([28, 28])

    plt.title("Sample: %d Label: %d" % (num, label))

    plt.imshow(image, cmap=plt.get_cmap("gray_r"))

    plt.show()


# display_sample(1233)

# create a placeholders for the images for trining data
input_images = tf.placeholder(tf.float32, shape=[None, 784])
target_labels = tf.placeholder(tf.float32, shape=[None, 10])

# topology
# creating variables to hold weight and biases between each run

hidden_nodes = 512
# setting a layer
input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))

hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
hidden_biases = tf.Variable(tf.zeros([10]))


# definin the connections
# matrix multiplication
input_layer = tf.matmul(input_images, input_weights)
hidden_laye = tf.nn.relu(input_layer + input_biases)
digit_weights = tf.matmul(hidden_laye, hidden_weights) + hidden_biases

# messure the correctens
loss_function = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=digit_weights, labels=target_labels)
)

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(digit_weights, 1), tf.argmax(target_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()


# training!!!!
for x in range(2000):
    batch = mnist.train.next_batch(100)
    optimizer.run(feed_dict={input_images: batch[0], target_labels: batch[1]})

    if (x + 1) % 100 == 0:
        print("Training epoch ->" + str(x + 1))
        print(
            "Accuracy ->"
            + str(
                accuracy.eval(
                    feed_dict={
                        input_images: mnist.test.images,
                        target_labels: mnist.test.labels,
                    }
                )
            )
        )

# testing
for x in range(100):
    x_train = mnist.test.images[x, :].reshape(1, 784)
    y_train = mnist.test.labels[x, :]

    label = y_train.argmax()
    prediction = sess.run(digit_weights, feed_dict={input_images: x_train}).argmax()

    if prediction != label:
        plt.title("Prediction %d label %d" % (prediction, label))
        plt.imshow(x_train.reshape([28, 28]), cmap=plt.get_cmap("gray_r"))
        plt.show()

