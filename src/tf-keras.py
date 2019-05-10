from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

(mnist_train_images, mnist_train_labels), (
    mnist_test_images,
    mnist_test_labels,
) = mnist.load_data()

# reshape the images as tf is spected to be
train_images = mnist_train_images.reshape(60000, 784)
test_images = mnist_test_images.reshape(10000, 784)
train_images = train_images.astype("float32")
test_images = test_images.astype("float32")
# normalize the images - rescale the data
train_images /= 255
test_images /= 255

train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
tests_labels = keras.utils.to_categorical(mnist_test_labels, 10)

# display a sample value
def display_sample(num):
    print(train_labels[num])
    # label back to number
    label = train_labels[num].argmax(axis=0)
    # reshaping th flat image
    image = train_images[num].reshape([28, 28])
    plt.title("sample %d  label %d" % (num, label))
    plt.imshow(image, cmap=plt.get_cmap("gray_r"))
    plt.show()


# display_sample(1235)
# defining the network
model = Sequential()  # the model
# 512 neurons 784 input
model.add(Dense(512, activation="relu", input_shape=(784,)))
# cuold be adding this options
# model.add(Dropout(0.2)) discard 20%
# model.add(Dense(512, activation="relu"))
# model.add(Dropout(0.2)) discard 20%
# final classification
model.add(Dense(10, activation="softmax"))

# Prints a string summary of the network
model.summary()

# check -> keras.io/optimizers
model.compile(
    loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
)

history = model.fit(
    train_images,
    train_labels,
    batch_size=100,
    epochs=50,
    verbose=2,
    validation_data=(test_images, tests_labels),
)

# evaluate the model accuracy
score = model.evaluate(test_images, tests_labels, verbose=0)
print("test loss-> ", score[0])
print("test accuracy-> ", score[1])


# whats goes wrong.. trouble to predict
# checks the first 1000 from the set
for x in range(1000):
    test_image = test_images[x, :].reshape(1, 784)
    predicted_cat = model.predict(test_image).argmax()
    label = tests_labels[x].argmax()
    if predicted_cat != label:
        plt.title("Prediction %d label %d" % (predicted_cat, label))
        plt.imshow(test_image.reshape([28, 28]), cmap=plt.get_cmap("gray_r"))
        plt.show()

