#   Created by Bayley King (Booth OW)
#   Python 3.7
#   Started on 12/16/2019
#   Github link 

import pandas as pd
'''
data = pd.read_csv("InData.csv", dtype=object)

for row in data.iterrows():
    print(row)

def search(choice,name,tier):
    # 
    print("search")


class Map():

    def __init__(self, data):
        self.team1 = data

'''

import pandas as pd
from os import system, name
from time import sleep
import timeit
import random
import numpy as np

def clear():
    _ = system('clear')


clear()
inData = pd.read_csv("InData.csv", dtype=object)
outData = pd.read_csv("OutData.csv",dtype=object)
players1 = ['T1P1','T1P2','T1P3','T1P4','T1P5','T1P6']
players2 = ['T2P1','T2P2','T2P3','T2P4','T2P5','T2P6']



def printPlayers(player,maps):
    for p in player:
        print('\t',inData[p][maps])

def newTeams(team1,team2,maps):
    print('Team 1:',team1)
    printPlayers(players1,maps)        
    print('Team 2:',team2)
    printPlayers(players2,maps)
    return team1,team2

def score(team1,team2,t1Score,t2Score,mode,maps):
    if t1Score > t2Score:
        outData[mode+' W'][maps]
        team2.Loss += 1
        #Team 1 won the map
    elif t2Score > t1Score:
        #Team 2 won the map
    else:
        #Draw

def mapScore(team1,team2,maps):
    if inData['MapType'][maps] == 'Control':

    elif inData['MapType'][maps] == 'Assualt':

    elif inData['MapType'][maps] == 'Hybrid':

    else:


team1 = inData['team1'][0]
team2 = inData['team2'][0]   
newTeams(team1,team2,0)
team1Score = team2Score =0

for maps in range(len(inData)):
    newTeam1 = inData['team1'][maps]
    newTeam2 = inData['team2'][maps]

    if newTeam1 != team1 or newTeam2 != team2:
        team1,team2 = newTeams(newTeam1,newTeam2,maps)

    if inData['MapType'][maps] == 'Final': # Final score entry
        print('\nFinal Score:',inData['Team1Score'][maps],inData['Team2Score'][maps])
        if inData['Team1Score'][maps] > inData['Team2Score'][maps]:
            team1Score += 1
        else:
            team2Score += 1
    else: # Normal Map. Calc score
        print(inData['MapType'][maps],'\n\t',inData['Map'][maps],'\n\t\t',inData['Team1Score'][maps],inData['Team2Score'][maps])

print('\nTeam 1 Score:',team1Score)
print('Team 2 Score:',team2Score)