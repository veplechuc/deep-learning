import tensorflow
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt

# load trainning and test samples
(mnist_train_images, mnist_train_labels), (
    mnist_test_images,
    mnist_test_labels,
) = mnist.load_data()


if K.image_data_format == "channels_first":
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 1, 28, 28)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28, 1)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")
# normalize the images - rescale the data
train_images /= 255
test_images /= 255

# convert labels into categorical in one-hot format
train_labels = tensorflow.keras.utils.to_categorical(mnist_train_labels, 10)
tests_labels = tensorflow.keras.utils.to_categorical(mnist_test_labels, 10)


#  display a sample value
def display_sample(num):
    print(train_labels[num])
    # label back to number
    label = train_labels[num].argmax(axis=0)
    # reshaping th flat image
    image = train_images[num].reshape([28, 28])
    plt.title("sample %d  label %d" % (num, label))
    plt.imshow(image, cmap=plt.get_cmap("gray_r"))
    plt.show()


display_sample(59999)

# set up CNN
# setup a secueltial model
# model = Sequential()
# # first layer
# model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
# # 64 3x3 kernels
# model.add(Conv2D(64, (3, 3), activation="relu"))
# # reduce by taking max of each 2x2 block
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # dropout to avoid overfitting
# model.add(Dropout(0.25))
# # flattern the result to one dimesion for passing into our final layer
# model.add(Flatten())
# # Hidden layer to learn with
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.25))
# # categorization from 0-9
# model.add(Dense(10, activation="softmax"))

# print(model.summary())

# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# history = model.fit(
#     train_images,
#     train_labels,
#     batch_size=32,
#     epochs=10,
#     verbose=2,
#     validation_data=(test_images, tests_labels),
# )

# score = model.evaluate(test_images, tests_labels, verbose=0)

# print("Test loss->", score[0])
# print("Test accuracy->", score[1])

