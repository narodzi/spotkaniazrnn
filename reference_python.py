import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.utils import to_categorical

# Załaduj zbiór danych MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizuj dane
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Konwertuj etykiety na wektory jednokrotnego wyboru (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Zbuduj model
model = Sequential([
    Input(shape=(28,28)),
    SimpleRNN(196),  # Warstwa SimpleRNN z 128 jednostkami
    Dense(10, activation='softmax')        # Warstwa wyjściowa z 10 neuronami (po jednej na każdą klasę)
])

# Skompiluj model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Wytrenuj model
history = model.fit(x_train, y_train, epochs=5, batch_size=100, validation_split=0.2)

# Oceń model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Wizualizacja historii trenowania
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
