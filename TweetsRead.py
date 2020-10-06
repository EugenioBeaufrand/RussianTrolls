import os
import csv
import json
import pandas as pd
from Project_Helper import ProjectPath


os.chdir(ProjectPath()+'/RegularTweets')
BigData = []



for filename in os.listdir(os.getcwd()):
  with open(filename,encoding='utf-8') as f:
   df = pd.DataFrame(json.loads(json.dumps(line,ensure_ascii=False).encode('utf-8')) for line in f)

   for row in range(df.shape[0]):

     if 'text' in json.loads(df.iloc[row][0]).keys() and 'user' in json.loads(df.iloc[row][0]).keys():
       Text = json.loads(df.iloc[row][0])['text']
       User = json.loads(df.iloc[row][0])['user']
       BigData.append([Text,User['name'],User['screen_name'],User['followers_count'],User['favourites_count'],User['statuses_count']])


BigData = pd.DataFrame(BigData)
BigData.columns = ['text','screen_name','name','follower_count','favorite_count','statuses_count']
BigData = BigData.dropna(axis=0,subset = ['text'], thresh=1)

os.chdir(ProjectPath+'/Data')
BigData.to_csv('NormalTweets.csv')

os.chdir(ProjectPath+'/RussianTweets')

RussianTweets = pd.read_csv('tweets.csv')
print(RussianTweets.shape)
RussianTweets = RussianTweets.dropna(axis=0, subset=['text'], thresh=1)
print(RussianTweets.shape)

with open('users.csv','r',encoding='UTF8') as File:
  FileRead = csv.reader(File)
  next(FileRead)
  Users = {}
  for row in FileRead:
    Users[int(row[0])] = [row[8],row[2],row[3],row[11],row[4]]

RussianTweetsAndUser = []

for tweet in range(RussianTweets.shape[0]):

  if RussianTweets.iloc[tweet][0] in Users.keys():
    RussianTweetsAndUser.append([RussianTweets.iloc[tweet][7].lower(),Users[RussianTweets.iloc[tweet][0]][0].lower(),Users[RussianTweets.iloc[tweet][0]][1].lower(),Users[RussianTweets.iloc[tweet][0]][2],Users[RussianTweets.iloc[tweet][0]][3],Users[RussianTweets.iloc[tweet][0]][4]])

  if tweet % 5000 == 0:
    print(tweet)

RussianTweetsAndUser = pd.DataFrame(RussianTweetsAndUser)
os.chdir(ProjectPath+'/Data')
RussianTweetsAndUser.to_csv('RussianTweets.csv')