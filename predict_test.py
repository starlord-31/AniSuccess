
import requests


url = 'http://localhost:9696/predict'

anime_id = 'xyz-123'
new_data_point = {
    'type': 'tv',
    'source': 'manga',
    'status': 'finished_airing',
    'rating': 'pg-13 - teens 13 or older',
    'studios': 'production_i.g',
    'genres': 'sports',
    'themes': 'school, team_sports',
    'producers': 'dentsu, mainichi_broadcasting_system, movic, toho_animation, shueisha, spacey_music_entertainment',
    'episodes': 25.0,
    'duration_minutes': 24.0,
    'synopsis_length': 1153
}

response = requests.post(url, json=new_data_point).json()
print(response)

if response['success']:
    print('Anime %s is predicted to be successful!' % anime_id)
else:
    print('Anime %s is not predicted to be successful.' % anime_id)