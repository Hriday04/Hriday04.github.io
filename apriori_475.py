# -*- coding: utf-8 -*-
"""Apriori_475.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GmCrXxJNSyOOlRNX7-Roeoo37DvL7Z6Y
"""

from google.colab import drive
drive.mount('/content/drive')

import json
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules



#json_file_path='/Users/hhraj/Downloads/spotify_million_playlist_dataset/data/mpd.slice.0-999.json'
#json_file_path = '/content/drive/Shareddrives/CS-351 Final/mpd.slice.0-999.json'
#json_path = '/content/drive/MyDrive/SpotifyML/sampleJSONs/mpd.slice.'

json_path = '/content/drive/MyDrive/SpotifyML/sampleJSONs/mpd.slice.'
json_list = ['0-999.json', '1000-1999.json', '2000-2999.json', '3000-3999.json',
             '4000-4999.json', '5000-5999.json', '6000-6999.json', '7000-7999.json']

data_dict = {}  # Initialize an empty dictionary to store the data

for json_file in json_list:
    json_file_path = json_path + json_file
    with open(json_file_path, 'r') as file:
        data_dict[json_file] = json.load(file)  # Store data with the file name as the key

# Now data_dict contains the data from each file with the file name as its key

#addie's stuff-- skip running
data_dict.keys()

def parse_playlist(playlist):
    return {
        "pid": playlist["pid"],
        "tracks": [
            {
                "track_uri": track["track_uri"]
            } for track in playlist["tracks"]
        ]
    }

# Initialize an empty list to store all parsed playlists
all_parsed_playlists = []

# Iterate over each file's data in the data_dict
for file_name, file_data in data_dict.items():
    # Process each playlist in the 'playlists' list of the current file
    parsed_playlists = [parse_playlist(playlist) for playlist in file_data["playlists"]]
    # Extend the main list with the parsed playlists from the current file
    all_parsed_playlists.extend(parsed_playlists)

# Now all_parsed_playlists contains the combined parsed data from all files

len(all_parsed_playlists)

#function to parse playlists
def parse_playlist(playlist):
    return {
        "pid": playlist["pid"],
        "tracks": [
            {
                "track_uri": track["track_uri"]
            } for track in playlist["tracks"]
        ]
    }

# Process each playlist in the 'playlists' list
#parsed_playlists = [parse_playlist(playlist) for playlist in data["playlists"]]

# Create a DataFrame from the transformed data
transformed_data = []
for playlist in all_parsed_playlists:
    playlist_dict = {}
    for track in playlist['tracks']:
        track_uri = track['track_uri']
        playlist_dict[track_uri] = 1
    transformed_data.append(playlist_dict)
df = pd.DataFrame(transformed_data)

df.head()
df.tail()



column_sums = df.sum()
threshold = 400
columns_to_drop = column_sums[column_sums < threshold].index
df = df.drop(columns=columns_to_drop)
df = df.fillna(0)
df = df.astype(bool)
#print(sorted(df.columns))
print(len(df.columns))
print(len(set(df.columns)))

frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)

# Generating association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(rules)
rules.to_csv('association_rules.csv', index=False)

def recommend_tracks(input_track_uris, rules, num_recommendations=10):
    recommended_tracks = set()
    for track_uri in input_track_uris:
        consequent_tracks = rules[rules['antecedents'] == frozenset([track_uri])]['consequents']
        for consequent in consequent_tracks:
            recommended_tracks.update(consequent)

    return list(recommended_tracks)[:num_recommendations]

#input list of track URIs
input_track_uris = ['spotify:track:0QsvXIfqM0zZoerQfsI9lm',"spotify:track:7BKLCZ1jbUBVqRi2FVlTVw","spotify:track:3a1lNhkSLSkpJE4MSHpDu9"]

# recc tracks based on the input list of track URIs
recommended_tracks = recommend_tracks(input_track_uris, rules, num_recommendations=10)

# Display the recommended tracks
print("Recommended Tracks:", recommended_tracks)
for track_uri in recommended_tracks:
    print(track_uri)

import subprocess

# Specify the package you want to install
package_name = "spotipy"

# Use subprocess to run the pip install command
subprocess.check_call(["pip", "install", package_name])
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id = '1c6f21ebb7fd40d88c81d8cdbc85b74e'
client_secret = '18ff2550a60c45cb8bf5037621efb6be'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

track  = sp.track('spotify:track:0QsvXIfqM0zZoerQfsI9lm')
song_name2 = track['name']
print(song_name2)

# Extract the song name from the track information
for track_uri in recommended_tracks:
    spotify_uri = track_uri
    track_info = sp.track(spotify_uri)
    song_name = track_info['name']
    artist_name = track_info['artists'][0]['name']
    track_info = sp.track(spotify_uri)
    print(f"Song Name: {song_name}, Artist: {artist_name}")

track  = sp.track('spotify:track:7BKLCZ1jbUBVqRi2FVlTVw')
song_name2 = track['name']
print(song_name2)

track2  = sp.track('spotify:track:3a1lNhkSLSkpJE4MSHpDu9')
song_name3 = track2['name']
print(song_name3)