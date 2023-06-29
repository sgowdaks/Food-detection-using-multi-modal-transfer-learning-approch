from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout

def lstm_model(vocab_size, embedding_dim, input_length):
    """
    LSTM Model
    """
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='softmax'))
    return model

def cnn_model(input_shape):
    """
    CNN Model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='softmax'))
    return model
