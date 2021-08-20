from mido import MidiFile, tempo2bpm, tick2second
from src.utils.create_mid_from_notes import create_mid_file_from_notes
from src.utils.definitions import Note

def calculate_beat_num(ticks_per_beat: int, delta_time: int):
    return delta_time / ticks_per_beat

class MidiParser:
    def __init__(self, file, create_test_file=True) -> None:
        mid = MidiFile(file, clip=True)

        self._mid = mid
        self._ticks_per_beat = mid.ticks_per_beat


        right = next((t for t in mid.tracks if t.name.lower() == 'right hand'), None)
        left = next((t for t in mid.tracks if t.name.lower() == 'left hand'), None)

        self._right_tack =right
        self._left_track = left

        self.left_note_list = self._parse_main_track(self._left_track)
        self.right_note_list = self._parse_main_track(self._right_tack)
        
        if create_test_file:
            create_mid_file_from_notes(self.right_note_list, self.left_note_list)

    
    def _parse_main_track(self, track):
        if track is None:
            return None
        note_list, temp_notes = [], {}
        current_time, note_idx = 0, 0

        for msg in track:
            if hasattr(msg, 'note'):
                str_msg = str(msg)
                current_time += msg.time
                if str_msg.startswith('note_on'):
                    temp_notes[msg.note] = {'tick_time': current_time,
                                            'pressed' : True,
                                            'list_idx': len(note_list)
                                            }
                    note_list.append(Note(note=msg.note, time=calculate_beat_num(self._ticks_per_beat, current_time) ))
                elif str_msg.startswith('note_off'):
                    note_dict = temp_notes.get(msg.note, None)
                    if note_dict is None or not note_dict["pressed"]:
                        continue
                    note_dict['pressed'] = False
                    list_idx, duration = note_dict['list_idx'], calculate_beat_num(self._ticks_per_beat,
                                                                                 current_time - note_dict['tick_time'])
                    note_list[list_idx].add_duration(duration)
                else:
                    print('ERROR') #Raise error
                temp_notes['note'] = msg.note
                note_idx += 1
        
        return note_list
        

if __name__ == "__main__":
    m = MidiParser('mid_test_files/CastlevaniaLevel1AS.mid')
