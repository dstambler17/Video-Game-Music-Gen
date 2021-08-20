import tensorflow as tf
from tensorflow import keras


def _RIGHT_HAND_MODEL(n_embedding_dims=5):
    '''
    Define the first right hand model
    '''
    model = keras.models.Sequential([
        keras.layers.Embedding(input_dim=84, output_dim=n_embedding_dims,
                            input_shape=[None]),
        keras.layers.Conv1D(64, kernel_size=2, padding="causal", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Conv1D(96, kernel_size=2, padding="causal", activation="relu", dilation_rate=2),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Conv1D(128, kernel_size=2, padding="causal", activation="relu", dilation_rate=4),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Conv1D(192, kernel_size=2, padding="causal", activation="relu", dilation_rate=8),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.LSTM(256, return_sequences=True),
        keras.layers.Dense(84, activation="softmax")
    ])
    return model

def _LEFT_HAND_MODEL(n_embedding_dims=5):
    '''
    Define the first left hand model
    '''
    model_left = keras.models.Sequential([
        keras.layers.Embedding(input_dim=69, output_dim=n_embedding_dims,
                            input_shape=[None]),
        keras.layers.Conv1D(64, kernel_size=2, padding="causal", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Conv1D(96, kernel_size=2, padding="causal", activation="relu", dilation_rate=2),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Conv1D(128, kernel_size=2, padding="causal", activation="relu", dilation_rate=4),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Conv1D(192, kernel_size=2, padding="causal", activation="relu", dilation_rate=8),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.LSTM(256, return_sequences=True),
        keras.layers.Dense(69, activation="softmax")
    ])
    return model_left

if __name__ == "__main__":
    model = _RIGHT_HAND_MODEL()
    model.summary()