import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.utils.definitions import Hand, Note
from src.utils.create_mid_from_notes import create_mid_file_from_notes
from src.utils.utils import normalize_note

def generate_notes_basic(model, seed_notes : list, length : int, hand : str):
    '''
    Basic logic to generate notes, always picks the best next note
    '''
    arpegio = tf.reshape(seed_notes, [1, -1])
    for note in range(length):
        #next_note = model.predict_classes(arpegio)[:1, -1:]
        next_note = np.argmax(model.predict(arpegio), axis=-1)[:1, -1:]
        arpegio = tf.concat([arpegio, next_note], axis=1)
    min_note = 21 if hand == Hand.Right.value else 24
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
    return arpegio


def generate_notes_varied(model, seed_notes : list, length : int, hand : str, temperature :int = 1):
    '''
    Logic to generate notes where the best stuff is not picked next
    Should use this logic over the above functions
    '''

    seed_notes = [normalize_note(note, hand) for note in seed_notes]  #First preprocess

    arpegio = tf.reshape(seed_notes, [1, -1])
    tf.cast(arpegio, tf.int32)
    for note in range(length):
        next_note_probas = model.predict(arpegio)[0, -1:]
        rescaled_logits = tf.math.log(next_note_probas) / temperature
        next_note = tf.random.categorical(rescaled_logits, num_samples=1)
        next_note = tf.cast(next_note, tf.int32)

        arpegio = tf.concat([arpegio, next_note], axis=1)
    min_note = 21 if hand == Hand.Right.value else 24
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
    return arpegio


def play_from_tensors(song_file_name : str, right_song_list : list, left_song_list : list = None):
    '''
    Converts generated lists to proper format
    And then plays music
    '''
    right_notes_to_play = [Note(note=note.numpy()) for note in right_song_list[0]]
    
    left_notes_to_play = None 
    if left_song_list is not None:
        left_notes_to_play = [Note(note=note.numpy()) for note in left_song_list[0]]

    create_mid_file_from_notes(right_notes_to_play, left_notes_to_play, song_file_name)


def generate_music_from_outputs(track_name, length, right_model, right_samples,
                                    left_model=None, left_samples=None):
    '''
    Main music generation function,
    Passes each hand model to generate_notes_varied to get output list
    '''
    right_song_list = generate_notes_varied(right_model, right_samples, length, 'right', temperature = 1.45)
    
    left_song_list = None
    if left_model is not None and left_samples is not None:
        left_song_list = generate_notes_varied(left_model, left_samples, length, 'left', temperature = 1.15)

    play_from_tensors('generated_music/%s.mid' % track_name, right_song_list, left_song_list)

if __name__ == "__main__":
    #Note, this data is already 'normalized'
    right_model = keras.models.load_model('src/models/saved_models/right_hand_vid_game_music_model.h5')
    sample_data = [39, 47, 39, 47, 39, 47, 39, 47, 40, 49, 40, 49, 44, 44, 42, 42, 40,
       40, 42, 42, 44, 44, 37, 37, 44, 44, 42, 42, 40, 40, 42, 42]
    
    left_model = keras.models.load_model('src/models/saved_models/left_hand_vid_game_music_model.h5')
    sample_data_left = [34, 42, 34, 42, 34, 42, 34, 42, 34, 42, 34, 42, 34, 42, 34, 42, 33, 41, 33, 41, 33, 41, 33, 41,
        33, 41, 33, 41, 33, 41, 33, 41]
    generate_music_from_outputs('gen_music_samples', 53, right_model, sample_data, left_model, sample_data_left)