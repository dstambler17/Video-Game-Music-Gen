# Video-Game-Music-Gen
Project that uses Recursive Neural Nets to generate video game music midi tracks


## Running the Project

#### Set up

Make sure you have the python path set up. Then run the following command:
```
export vid_game_music_ml_data='REPLACE ME WITH THE DESIRED PATH'
```

#### Downloading the data
All data comes from this site: https://www.vgmusic.com/music/other/miscellaneous/piano/
Credit to the site owner and to all the original composers for the data

To download the data, run the following commands in order
```
python3 src/data_scripts/download_piano_mid_files.py
python3 src/data_scripts/organize_data_files.py
```

### Training
If you wish to train the models, run the following
```
python3 src/train_models.py
```

### Generating music
To generate some midi samples,
run the following. Using the beginning of some midi tracks
in the 'test' data set, this will generate several tracks
```
python3 src/main_process.py
```
The output should appear in the 'generated music folder'
Then just open those files and give them a listen




