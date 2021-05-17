from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype("float32")
x_train /= 255
y_train = to_categorical(y_train, num_classes=10)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype("float32")
x_test /= 255
y_test = to_categorical(y_test, num_classes=10)

img_shape = (28, 28, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=img_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=img_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(),
    metrics=["acc"]
)

model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=15,
    steps_per_epoch=200
)

model.evaluate(x_test, y_test)
model.save("./models/detect.h5")
