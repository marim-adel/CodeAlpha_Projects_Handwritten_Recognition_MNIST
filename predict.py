import tensorflow as tf, numpy as np
def load_model(path='models/mnist_cnn_tf'):
    return tf.keras.models.load_model(path)
def predict_image(model, img_array):
    # img_array: numpy array 28x28 or (28,28,1) scaled 0-1
    import numpy as np
    arr = np.array(img_array).astype('float32')/255.0
    if arr.ndim==2:
        arr = np.expand_dims(arr, -1)
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)
    return int(preds.argmax(axis=1)[0]), preds.max()
