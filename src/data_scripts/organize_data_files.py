import os
from shutil import copyfile
from src.utils.load_midi_file import MidiParser
import numpy as np


def filter_data_files(dir_name, hand='right'):
    '''
    Checks that the tracks contain between 100-2500 tracks,
    and that they contain only right/left hand piano tracks
    '''
    true_song_list = []
    for idx, f in enumerate(os.listdir(dir_name)):
        mp = MidiParser(dir_name + '/' + f, create_test_file=False)
        num_left_notes = 0 if mp.left_note_list is None else len(mp.left_note_list)
        num_right_notes = 0 if mp.right_note_list is None else len(mp.right_note_list)

        if hand == 'right':
            if num_right_notes >= 100 and num_right_notes <= 2000:
                true_song_list.append((f, num_right_notes))
        else:
            if num_left_notes >= 100 and num_left_notes <= 2000:
                true_song_list.append((f, num_left_notes))
    print(len(true_song_list)) #250 total records of length 100 to 2500
    return true_song_list


def train_val_test_split(filtered_tracks):
    '''
    Splits tracks into train val and test proportionally by length
    Goes by 100 at a time
    '''

    train, val, test = [], [], []
    for i in range(100, 2000, 100):
        start, end = i, i + 99
        subset = list(filter(lambda x: x[1] >=start and x[1] <= end, filtered_tracks))
        subset = [x[0] for x in subset]
        if len(subset) >= 3:
            train_sub, val_sub, test_sub = np.split(subset, 
                                    [int(.9 * len(subset)), int(.98 * len(subset))])
            
            train += list(train_sub)
            val += list(val_sub)
            test += list(test_sub)
    import random
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test
    


def save_filtered_tracks(old_dir_name, new_dir_name, train, val, test, make_new_dir=False, split_version=''):
    '''
    Writes these files to the final data se folder
    Saves different files to the val, test and train set
    '''
    if make_new_dir:
        #os.mkdir(new_dir_name)
        os.mkdir(new_dir_name + '/train' + split_version)
        os.mkdir(new_dir_name + '/val' + split_version)
        os.mkdir(new_dir_name + '/test' + split_version)

    for idx, file_track in enumerate(train):
        sub_folder = '/train%s/' % split_version
        copyfile(old_dir_name + '/' + file_track, new_dir_name + sub_folder + file_track)
    
    if val is None or test is None:
        return

    for idx, file_track in enumerate(val):
        sub_folder = '/val%s/' % split_version
        copyfile(old_dir_name + '/' + file_track, new_dir_name + sub_folder + file_track)

    for idx, file_track in enumerate(test):
        sub_folder = '/test%s/' % split_version
        copyfile(old_dir_name + '/' + file_track, new_dir_name + sub_folder + file_track)


if __name__ == "__main__":
    DIRECTORY_NAME = 'data/dataset_midi_samples'
    filtered_tracks = filter_data_files(DIRECTORY_NAME, hand='left')
    print(len(filtered_tracks))
    train, val, test = train_val_test_split(filtered_tracks)
    save_filtered_tracks(DIRECTORY_NAME, 'data/dataset_midi_final', train, val, test, True, "_left_v1")
   