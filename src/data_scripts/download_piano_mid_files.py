import requests
import re
from bs4 import BeautifulSoup
from tensorflow import keras
from pathlib import Path
import subprocess

URL = 'https://www.vgmusic.com/music/other/miscellaneous/piano/'


def _convert_to_megabytes(bytes: int):
    '''
    1 Bytes = 9.537×10-7 Megabytes
    '''
    return 0.0000009537 * bytes

#Note, looks like there are 721 files total
def count_total_byte_size(text: str):
    bytes_re = re.compile('\d+\sbytes', flags=re.IGNORECASE)
    hits = bytes_re.findall(text)
    
    total_bytes = 0
    for hit in hits:
        byte_num = int(hit.split(' ')[0])
        total_bytes += byte_num
    print("Size of all combined files is %f megabytes" % _convert_to_megabytes(total_bytes))


def get_track_list(res: str):
    track_re = re.compile('href=\"(.+\.mid)\"', flags=re.IGNORECASE)
    hits = track_re.findall(res)

    print("Total tracks = %d" % len(hits))
    return hits

def download_file(filename):
    filepath = keras.utils.get_file(filename,
                                    URL + filename,
                                    cache_subdir="datasets/dataset_midi_samples",
                                    extract=True)
    print(filepath)
    return filepath


def download_dataset():
    r = requests.get(URL)
    res = r.text
    count_total_byte_size(res)
    track_list = get_track_list(res)
    for track in track_list:
        file_path = download_file(track)
        midi_dir = Path(file_path).parent
    
    if midi_dir is not None:
        subprocess.call("mv %s ." % midi_dir, shell=True)


if __name__ == "__main__":
    download_dataset()