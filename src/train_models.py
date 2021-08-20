import tensorflow as tf
from tensorflow import keras
from src.preprocess.preprocess import main_preprocess, count_number_unique_notes
from src.models.models import _RIGHT_HAND_MODEL, _LEFT_HAND_MODEL
from src.utils.definitions import Hand


def train_model(model, train_set, val_set, epoch=5):
    '''
    Trains and saves the model
    '''
    optimizer = keras.optimizers.Nadam(lr=1e-3)
    model.compile(loss="sparse_categorical_crossentropy",
                     optimizer=optimizer,
                    metrics=["accuracy"])
    model.fit(train_set, epochs=epoch, validation_data=val_set),


def evaluate_model(model, test_set):
    '''
    Evaluates the model on the test set
    '''
    model.evaluate(test_set)


def save_model(model, model_path : str):
    '''
    Saves model
    '''
    model.save(model_path)

def handle_main_training_process(hand):
    '''
    Gets preprocessed data, trains the model
    Then calls the evaluation function
    '''
    if hand == Hand.Right.value:
        train_dir = 'train_right_v_recent'
        val_dir = 'val_right_v_recent'
        test_dir = 'test_right_v_recent'
    else:
        train_dir = 'train_left_v1'
        val_dir = 'val_left_v1'
        test_dir = 'test_left_v1'

    n_notes = count_number_unique_notes(hand)
    train_set= main_preprocess(hand, train_dir, shuffle_buffer_size=1000)
    val_set = main_preprocess(hand, val_dir)
    test_set = main_preprocess(hand, test_dir)

    model = _RIGHT_HAND_MODEL() if hand == Hand.Right.value else _LEFT_HAND_MODEL()

    train_model(model, train_set, val_set)
    
    #NOTE: To save the model, uncomment and change the path
    #save_model(model, 'src/models/saved_models/%s_hand_video_game_music_tune_model.h5' % hand)
    evaluate_model(model, test_set)
    
if __name__ == "__main__":
    handle_main_training_process('right')
    handle_main_training_process('left')