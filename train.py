import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import os
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
num_classes = 10
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
]
model.fit(x_train, y_train_cat, epochs=6, batch_size=256, validation_split=0.1, callbacks=callbacks)
eval_res = model.evaluate(x_test, y_test_cat, verbose=0)
print('Test loss, acc:', eval_res)
os.makedirs('models', exist_ok=True)
model.save('models/mnist_cnn_tf')
# save a small sample mapping
import joblib
joblib.dump({'classes': list(range(10))}, 'models/mnist_meta.joblib')
print('Saved model to models/mnist_cnn_tf')
