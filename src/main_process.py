import os
import tensorflow as tf
from tensorflow import keras
from statistics import mean
import numpy as np
from src.utils.definitions import Hand, Note
from src.music_gen.generate_music import generate_music_from_outputs

from src.utils.definitions import DIRECTORY_BASE_NAME
from src.utils.utils import load_data_set

def get_qualified_tracks_from_midi_dir(sub_dir: str):
    '''
    Returns list of tracks with both hands
    from midi files
    '''
    qualified_files = []

    files_dir = os.path.join(DIRECTORY_BASE_NAME, sub_dir)
    double_list = load_data_set(files_dir, None, True)
    for (right, left) in double_list:
        if not right or not left:
            continue
        qualified_files.append((right, left))

    return qualified_files


def get_average_tempo_dur(note_list):
    '''
    Calcs the average temp and duration
    '''

    times = np.array([note.time for note in note_list])
  
    # Calculating difference list
    time_diff_list = np.diff(times)
    avg_time_increments = mean(time_diff_list)
    if 'duration' not in note_list[0].__dict__:
         avg_duration = None
         return avg_duration, avg_time_increments
    
    avg_duration = mean([note.duration for note in note_list])
    
    return avg_duration, avg_time_increments

def main_process_batch(sub_dir: str, right_model_path: str, left_model_path: str = None):
    '''
    This functions generates a batch of outputs from a batch of inputs.
    1) It gets a list (from the test set)
    of Tracks that have notes for right and left hand

    2) Generates music for these examples
    '''
    both_hand_lists = get_qualified_tracks_from_midi_dir(sub_dir)
    
    processed = []
    for right, left in both_hand_lists:
        new_notes_left = [n.note for n in left]
        new_notes_right = [n.note for n in right]
        processed.append((new_notes_right, new_notes_left))
    
    right_model = keras.models.load_model( right_model_path)

    left_model = None
    if left_model_path: #left model can be optional
        left_model = keras.models.load_model( left_model_path)
    

    #TEMP: Testing using average dur/time
    #JUST TESTS WHAT HAPPENS WHEN YOU PROCESS BOTH TEMPOS
    #average_tempo_right, average_dur_right = get_average_tempo_dur(both_hand_lists[12][0])
    #average_tempo_left, average_dur_left = get_average_tempo_dur(both_hand_lists[12][1])
    #generate_music_from_outputs('batch_samples_dur_test', 100, right_model,
    #                             processed[12][0][0:50], left_model, processed[12][1][0:50])
    
    for idx, (right, left) in enumerate(processed[:5]):
        generate_music_from_outputs('batch_samples_%d' % idx, 100, right_model,
                                 right[0:50], left_model, left[0:50])


if __name__ == "__main__":
    RIGHT_MODEL_PATH = 'src/models/saved_models/right_hand_vid_game_music_model.h5'
    LEFT_MODEL_PATH = 'src/models/saved_models/left_hand_vid_game_music_model.h5'
    main_process_batch('test_right_v_recent', RIGHT_MODEL_PATH, LEFT_MODEL_PATH)


