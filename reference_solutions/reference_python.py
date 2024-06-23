import numpy as np
import matplotlib.pyplot as plt
import keras
import psutil
import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.utils import to_categorical

class MemoryAndTimeCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        print(f"Epoch {epoch + 1}")
        print(f"Time: {elapsed_time:.2f} seconds")
        print_memory_usage()
        train_loss, train_accuracy = logs.get('loss'), logs.get('accuracy')
        val_loss, val_accuracy = logs.get('val_loss'), logs.get('val_accuracy')
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}")
        print()

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024**2:.2f} MiB")

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
optimizer = keras.optimizers.Adam(learning_rate=0.015)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Wytrenuj model
model.fit(x_train, y_train, epochs=5, batch_size=100, validation_split=0.2, callbacks=[MemoryAndTimeCallback()])

# Oceń model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
