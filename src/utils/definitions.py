import os
from enum import Enum

DIRECTORY_BASE_NAME = os.environ.get('vid_game_music_ml_data')

class Hand(Enum):
    Right = 'right'
    Left = 'left'

class Note:
    def __init__(self, **kwargs) -> None:
        self.channel = 0
        self.__dict__.update(kwargs)
    
    def add_duration(self, duration: int) -> None:
        self.__dict__.update({'duration' : duration})
    
    def __str__(self) -> str:
        return 'note=%d, duration=%f, time=%f' % (self.note, self.duration, self.time)