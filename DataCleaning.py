from Project_Helper import ProjectPath
import pandas as pd
import os


os.chdir(ProjectPath()+'/Data')

NormalData = (pd.read_csv('NormalTweets.csv')).drop((pd.read_csv('NormalTweets.csv')).columns[0],axis='columns')
NormalData = NormalData.drop_duplicates()
NormalData.columns = ['tweet','screen_name','name','follower_count','favorite_count','statuses_count']
NormalData['class'] = 0
NormalData = NormalData.dropna()
print(NormalData.shape)

RussianData = (pd.read_csv('RussianTweets.csv')).drop((pd.read_csv('RussianTweets.csv')).columns[0],axis='columns')
RussianData = RussianData.drop_duplicates()
RussianData.columns = ['tweet','screen_name','name','follower_count','favorite_count','statuses_count']
RussianData['class'] = 1
RussianData = RussianData.dropna()
print(RussianData.shape)

NormalSample = NormalData.sample(n = 100000)
RussianData = RussianData.sample(n = 25000)

DataSet = pd.concat([NormalSample,RussianData])

DataSet.to_csv('DataSet.csv',index=False)

