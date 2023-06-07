import requests
import time
import urllib.parse
import pandas as pd

# borderlands 3
# warframe
# division 2
# outriders
# titanfall 2
# destiny 2

APP_ID = ['397540', '1245620', '1151340', '680420', '1237970', '1085660']
STEAM_URL = "https://store.steampowered.com"

reviews = []
positives = []
votes = []
scores = []

days = []
months = []
dates = []
times = []
years = []

games = []

for app in APP_ID:
    for i in range(0, 200):
        if i == 0:
            cursor = '*'
            
        data = requests.get(STEAM_URL + f"/appreviews/{app}?json=1&cursor={cursor}&filter=recent&language=english&num_per_page=100").json()
        
        for review in data['reviews']:
            if app == '397540':
                games.append('Borderlands 3')
            elif app == '1245620':
                games.append('Elden Ring')
            elif app == '1151340':
                games.append('Fallout 76')
            elif app == '680420':
                games.append('Outriders')
            elif app == '1237970':
                games.append('Titanfall 2')
            elif app == '1085660':
                games.append('Destiny 2')
            reviews.append(review['review'])
            
        for positive in data['reviews']:
            positives.append(1 if positive['voted_up'] else 0)
            
        for vote in data['reviews']:
            votes.append(vote['votes_up'])
            
        for score in data['reviews']:
            scores.append(score['weighted_vote_score'])
            
        for datetime in data['reviews']:
            x = time.ctime(datetime['timestamp_created'])
            x = x.split()
            
            days.append(x[0])
            months.append(x[1])
            dates.append(x[2])
            times.append(x[3])
            years.append(x[4])
            
        
        cursor = urllib.parse.quote(data['cursor'])
        print(cursor)
        i += 1
      
        
# RAW DATAFRAME
raw_df = pd.DataFrame({'DATE': dates, 'DAY': days, 'MONTH' : months, 'YEAR' : years, 'TIME' : times, 'GAME' : games, 'RAW' : reviews, 'POSITIVE' : positives, 'VOTES' : votes, 'SCORE' : scores})
raw_df.to_csv('raw_df.csv', index=False)