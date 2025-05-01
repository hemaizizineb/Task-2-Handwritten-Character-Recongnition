# main.py
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from ocr_model import build_ocr_model
from utils import load_and_preprocess_images

def main():
    """
    Train and evaluate an optical character recognition (OCR) model.

    Returns:
        None
    """
    # Load and preprocess training data
    X_train, y_train = load_and_preprocess_images('./src/data/training/')
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Load and preprocess testing data
    X_test, y_test = load_and_preprocess_images('./src/data/testing/')
    y_test = le.transform(y_test)

    # Build and compile the model
    model = build_ocr_model()
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size=16, epochs=10)

    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy:.2f}')

if __name__ == '__main__':
    main()