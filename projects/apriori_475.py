# -*- coding: utf-8 -*-
"""Apriori_475.ipynb"""

from google.colab import drive
import json
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import subprocess
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the JSON files
json_path = '/content/drive/MyDrive/SpotifyML/sampleJSONs/mpd.slice.'
json_list = ['0-999.json', '1000-1999.json', '2000-2999.json', 
             '3000-3999.json', '4000-4999.json', '5000-5999.json', 
             '6000-6999.json', '7000-7999.json']

# Load all JSON files into a dictionary
data_dict = {}
for json_file in json_list:
    with open(json_path + json_file, 'r') as file:
        data_dict[json_file] = json.load(file)

# Function to parse playlists
def parse_playlist(playlist):
    return {
        "pid": playlist["pid"],
        "tracks": [{"track_uri": track["track_uri"]} for track in playlist["tracks"]]
    }

# Parse all playlists from the data
all_parsed_playlists = []
for file_data in data_dict.values():
    all_parsed_playlists.extend([parse_playlist(playlist) for playlist in file_data["playlists"]])

# Transform the data into a DataFrame
transformed_data = []
for playlist in all_parsed_playlists:
    playlist_dict = {track['track_uri']: 1 for track in playlist['tracks']}
    transformed_data.append(playlist_dict)
df = pd.DataFrame(transformed_data).fillna(0).astype(bool)

# Drop columns below a certain threshold
threshold = 400
df = df.drop(columns=df.columns[df.sum() < threshold])

# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules.to_csv('association_rules.csv', index=False)

# Function to recommend tracks based on association rules
def recommend_tracks(input_track_uris, rules, num_recommendations=10):
    recommended_tracks = set()
    for track_uri in input_track_uris:
        consequents = rules[rules['antecedents'] == frozenset([track_uri])]['consequents']
        for consequent in consequents:
            recommended_tracks.update(consequent)
    return list(recommended_tracks)[:num_recommendations]

# Example input track URIs
input_track_uris = [
    'spotify:track:0QsvXIfqM0zZoerQfsI9lm',
    'spotify:track:7BKLCZ1jbUBVqRi2FVlTVw',
    'spotify:track:3a1lNhkSLSkpJE4MSHpDu9'
]

# Get recommended tracks
recommended_tracks = recommend_tracks(input_track_uris, rules, num_recommendations=10)

# Install the Spotipy package if not already installed
subprocess.check_call(["pip", "install", "spotipy"])

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret))

# Display recommended tracks with song and artist names
print("Recommended Tracks:")
for track_uri in recommended_tracks:
    track_info = sp.track(track_uri)
    song_name = track_info['name']
    artist_name = track_info['artists'][0]['name']
    print(f"Song Name: {song_name}, Artist: {artist_name}")
