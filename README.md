# RecSysSpotify



## STEP 1: Obtaining The Data
1. Download Spotify's official [dataset](recsys-challenge.spotify.com/dataset) and place the 'data' folder into the root folder of the project. 
2. save "challenge_set.json" into "utils_spotify" folder

## STEP 2: CREATE DICTIONARY AND MODELS used in the final recommender function
RUN:
+ 1_MPD_id_trackuri_maker.ipynb
+ 2_common_titles_and_tracks_maker.ipynb
+ 3_MPD_line_sentences_maker.ipynb
+ 4_word2vec_tracks.ipynb	
+ 5_word2vec_titles.ipynb

-- trained models and dictionaries can be downloaded from [here] 

## STEP 3: 
1. RUN all notebook "create_final_submission.ipynb" to create submission


### License
Usage of the Million Playlist Dataset is subject to these 
[license terms](https://recsys-challenge.spotify.com/license)
