import os
from src.utils.definitions import Hand
from src.utils.load_midi_file import MidiParser


def load_data_set(dir_name : str, hand: str, return_both : bool = False):
    '''
    Loads in all the data from a given directory
    TODO: Probably move this to utils and update imports
    '''
    note_list = []
    for idx, f in enumerate(os.listdir(dir_name)):
        mp = MidiParser(dir_name + '/' + f, create_test_file=False)
        left_notes = mp.left_note_list
        right_notes = mp.right_note_list

        if return_both: #Special case, for when you want to load both notes
            note_list.append((right_notes, left_notes))
        elif hand == Hand.Right.value:
            note_list.append(right_notes)
        else:
            note_list.append(left_notes)
    return note_list


def normalize_note(note: int, hand: str):
    #TODO: Move this to utils and adjust imports
    '''
    From analyzing the data
    right hand min note = 21,
    right hand max note = 103

    left hand min note = 24
    left hand max note = 91

    We want to change the data so that the min value is 1 and the max is
    either 103 - 20, aka 83 for the right hand
    or 91 - 23, aka 68 for the left hand
    '''

    min_val = 21 if hand == Hand.Right.value else 24
    min_val -= 1
    return note - min_val