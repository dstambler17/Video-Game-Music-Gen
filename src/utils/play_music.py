#Wave equation: g(f) = A sin(2πft)
#Relationship notes are tuned relative to one another: 
#   note_freq = base_freq * 2^(n/12)
#   n is the number of notes away from the base note
#Credit: https://towardsdatascience.com/mathematics-of-music-in-python-b7d838c84f72

import numpy as np
from scipy.io.wavfile import write

from IPython.display import Audio


samplerate = 44100 #Frequecy in Hz

def get_wave(freq, duration=0.5):
    '''
    Function takes the "frequecy" and "time_duration" for a wave 
    as the input and returns a "numpy array" of values at all points 
    in time
    '''
    
    amplitude = 4096
    t = np.linspace(0, duration, int(samplerate * duration))
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    
    return wave

def test_and_plot_get_wave():
    # To get a 1 second long wave of frequency 440Hz
    test_freq = 440
    a_wave = get_wave(test_freq, 1)

    #wave features
    print(len(a_wave)) # 44100
    print(np.max(a_wave)) # 4096
    print(np.min(a_wave)) # -4096

    import matplotlib.pyplot as plt
    plt.plot(a_wave[0:int(samplerate/test_freq)])
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.show()



def get_piano_notes():
    '''
    Returns a dict object for all the piano 
    note's frequencies
    '''
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    BASE_FREQ = 261.63 #Frequency of Note C4
    
    note_freqs = {octave[i]: BASE_FREQ * pow(2,(i/12)) for i in range(len(octave))}        
    note_freqs[''] = 0.0 # silent note
    
    return note_freqs
  

import numpy as np

def get_song_data(music_notes):
    '''
    Function to concatenate all the waves (notes)
    '''
    note_freqs = get_piano_notes() # Function that we made earlier
    song = [get_wave(note_freqs[note]) for note in music_notes.split('-')]
    song = np.concatenate(song)
    NOTE_DURATION = 60
    song = np.round(NOTE_DURATION * song) / NOTE_DURATION
    return song



if __name__ == "__main__":
    #test_and_plot_get_wave()
    # To get the piano note's frequencies
    note_freqs = get_piano_notes() #lower cases represent the black keys (Sharps))
    print(note_freqs)
    music_notes = 'C-C-G-G-A-A-G--F-F-E-E-D-D-C--G-G-F-F-E-E-D--G-G-F-F-E-E-D--C-C-G-G-A-A-G--F-F-E-E-D-D-C'
    data = get_song_data(music_notes)
    #data = data * (16300/np.max(data)) #adjust amplitude (optional thing)

    #Wrtie to scipy array to create music file

    write('test.wav', samplerate, data.astype(np.int16))
    Audio(data, rate=samplerate)


    

