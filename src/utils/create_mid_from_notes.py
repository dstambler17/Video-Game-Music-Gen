from midiutil.MidiFile import MIDIFile
from heapq import heapify, heappop, heappush

STANDARD_TIMESTAMP_INCREASE = 0.25 #These values are temporary and just from playing around
STANDARD_DURATION = 0.235417

def _create_mid_track(mf: MIDIFile, note_list : list, 
                        time_step_increase : float, 
                        duration : float, is_right_hand : bool = True):
    track = 0 if is_right_hand else 1
    time, channel, volume = 0, 0, 100

    track_name = "right hand" if is_right_hand else "left hand"
    mf.addTrackName(track, time, track_name)
    mf.addTempo(track, time, 120)

    #note_list.sort(key = lambda x: x.time)
    for idx, note in enumerate(note_list):
        mf.addNote(track, channel, int(note.note), time, duration, volume) #time, dur, volume
        time = time + time_step_increase



def create_mid_file_from_notes(right_list: list, left_list: list, file_name : str ='test.mid',
                            time_step_increase : float = STANDARD_TIMESTAMP_INCREASE, duration : float =STANDARD_DURATION):
    num_tracks = 2 if left_list is not None else 1
    mf = MIDIFile(num_tracks)
    
    _create_mid_track(mf, right_list, 
                        time_step_increase=time_step_increase, 
                        duration = duration)

    if left_list is not None:
        _create_mid_track(mf, left_list, 
                            time_step_increase=time_step_increase,
                            duration = duration,
                            is_right_hand= False)


    # write it to disk
    with open(file_name, 'wb') as outf:
        mf.writeFile(outf)