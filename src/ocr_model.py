# ocr_model.py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_ocr_model():
    """
    Build a convolutional neural network (CNN) model for optical character recognition (OCR).

    Returns:
        model: A Sequential Keras model instance.
    """
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(36, activation='softmax'))
    return model