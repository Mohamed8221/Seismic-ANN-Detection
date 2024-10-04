from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten

def build_model(input_shape):
    """Build a neural network model."""
    model = Sequential()

    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
