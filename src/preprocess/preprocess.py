import os
import tensorflow as tf
from tensorflow import keras
from src.utils.definitions import Hand, Note, DIRECTORY_BASE_NAME
from src.utils.utils import load_data_set, normalize_note



def count_number_unique_notes(hand: str):
    """
    Function for data analysis,
    gets all the different values and shows min/max notes
    Also returns length of all the notes
    """
    train_dir = 'train' if hand == 'right' else 'train_left_v1'
    val_dir = 'val' if hand == 'right' else 'val_left_v1'
    test_dir = 'test' if hand == 'right' else 'test_left_v1'

    val_notes = handle_preprocess(hand, val_dir)
    train_notes = handle_preprocess(hand, train_dir)
    test_notes  = handle_preprocess(hand, test_dir)
    
    notes = set()
    for note_list in (val_notes, train_notes, test_notes):
        for nl in note_list:
            for note in nl:
                notes.add(note)

    n_notes = len(notes)
    min_note = min(notes - {0})
    max_note = max(notes)

    print("PRINTING SET OF NOTES FOR %s HAND" % (hand.upper()))
    print(notes, min_note, max_note)
    return n_notes


def note_to_list(track_list, hand: str):
    '''
    Converts the list of lists of notes to just the
    list of lists of numerical notes
    '''
    new_notes = [[normalize_note(n.note, hand) for n in note_list] for note_list in track_list]
    return new_notes


def handle_preprocess(hand : str, sub_dir: str):
    '''
    Handles the pre model processing of the data before batching
    '''
    files_dir = os.path.join(DIRECTORY_BASE_NAME, sub_dir)
    print(files_dir)
    track_list = load_data_set(files_dir, hand)
    music_notes = note_to_list(track_list, hand)
    return music_notes


def create_target(batch):
    '''
    Helper function, used to split the data into features and labels
    '''
    X = batch[:, :-1]
    Y = batch[:, 1:] # predict next note at each step
    return X, Y


def batch_to_windows(data, batch_size=32, shuffle_buffer_size=None,
                 window_size=32, window_shift=16, cache=True):
    '''
    Converts data to the dataset object. Breaks that data into windows with
    X, y data, shuffles, batches them, and lastly prefetches the results
    '''

    def batch_window(window):
        return window.batch(window_size + 1)
    
    def to_windows(data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.window(window_size + 1, window_shift, drop_remainder=True)
        return dataset.flat_map(batch_window)

    data = tf.ragged.constant(data, ragged_rank=1)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.flat_map(to_windows) #Need to create two flatmaps due to the dataset being a list of lists

    #cache if cache enabled
    if cache:
        dataset = dataset.cache()

    #shuffle if shuffle enabled
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(create_target)

    return dataset.prefetch(1)


def main_preprocess(hand : str, sub_dir: str, batch_size : int =32, shuffle_buffer_size : int =None,
                 window_size: int = 32, window_shift: int = 16, cache: bool =True):
    '''
    Main preprocess function, gets the data ready
    to be fed to the model
    '''
    music_notes = handle_preprocess(hand, sub_dir)
    dataset = batch_to_windows(music_notes, 
                                batch_size=batch_size,
                                shuffle_buffer_size=shuffle_buffer_size,
                                window_size=window_size,
                                window_shift=window_shift,
                                cache=cache
                               )
    return dataset


if __name__ == "__main__":
   print(count_number_unique_notes('right'))
   #count_number_unique_notes('left')
   #main_preprocess('right' , 'val')