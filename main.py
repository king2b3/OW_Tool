#   Created by Bayley King (Booth OW)
#   Python 3.7
#   Started on 12/16/2019
#   Github link 

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
from tabulate import tabulate

def clear():
    _ = system('clear')


clear()
toPrint = []
inData = pd.read_csv("InData.csv", dtype=object)
outData = pd.read_csv("OutData.csv",dtype=object,index_col=0)
#players1 = ['T1P1','T1P2','T1P3','T1P4','T1P5','T1P6']
#players2 = ['T2P1','T2P2','T2P3','T2P4','T2P5','T2P6']

'''
def printPlayers(player,maps):
    for p in player:
        toPrint.append(['  ',inData[p][maps]])
def newTeams(team1,team2,maps):
    toPrint.append(['Team 1:',team1])
    printPlayers(players1,maps)        
    toPrint.append(['Team 2:',team2])
    printPlayers(players2,maps)
    return team1,team2
'''
def increaseScore(teamW,teamL,mode,Map,score1,score2,final):
    if final:
        outData.loc[teamW,'Game W'] = int(outData.loc[teamW,'Game W']) + 1
        outData.loc[teamL,'Game L'] = int(outData.loc[teamL,'Game L']) + 1

    else:
        #print(team1,mode,Map)
        for col in [mode+' W',Map+' W','Map W']:
            #print(mode)
            #print(Map)

            outData.loc[teamW,col] = int(outData.loc[teamW,col]) + 1
        for col in [mode+' L',Map+' L','Map L']:
            outData.loc[teamL,col] = int(outData.loc[teamL,col]) + 1
        for t,s in zip([teamW,teamL],[score1,score2]):
            outData.loc[t,mode+' M S'] = int(outData.loc[t,mode+' M S']) + int(s)


def score(team1,team2,t1Score,t2Score,mode,final,maps=None):
    if t1Score > t2Score:
        #print(team1,team2,t1Score,t2Score,mode,maps)
        increaseScore(team1,team2,mode,maps,t1Score,t2Score,final)
        #Team 1 won the map
    elif t2Score > t1Score:
        #Team 2 won the map
        increaseScore(team2,team1,mode,maps,t2Score,t1Score,final)
    else:
        if final:
            for t in [team1,team2]:
                outData.loc[t,'Game T'] = int(outData.loc[t,'Game T']) + 1
        else:
            for t in [team1,team2]:
                print(team1,team2,mode,maps,t1Score,t2Score,final)
                for col in [mode+' T',maps+' T','Map T']:
                    outData.loc[t,col] = int(outData.loc[t,col]) + 1
            for t,s in zip([team1,team2],[t1Score,t2Score]):
                outData.loc[t,mode+' M S'] = int(outData.loc[t,mode+' M S']) + int(s)
            

def main():

    team1Score = team2Score =0

    for maps in range(len(inData)):
        newTeam1 = inData['team1'][maps]
        newTeam2 = inData['team2'][maps]

        #if newTeam1 != team1 or newTeam2 != team2:
        #    team1,team2 = newTeams(newTeam1,newTeam2,maps)

        if inData['MapType'][maps] == 'Final': # Final score entry
            toPrint.append(['\nFinal Score:',None,inData['Team1Score'][maps],inData['Team2Score'][maps]])
            score(newTeam1,newTeam2,inData['Team1Score'][maps],inData['Team2Score'][maps],inData['MapType'][maps],True)

            if inData['Team1Score'][maps] > inData['Team2Score'][maps]:
                team1Score += 1
            else:
                team2Score += 1
        else: # Normal Map. Calc score
            score(newTeam1,newTeam2,inData['Team1Score'][maps],inData['Team2Score'][maps],inData['MapType'][maps],False,inData['Map'][maps])
            toPrint.append([inData['MapType'][maps],inData['Map'][maps],inData['Team1Score'][maps],inData['Team2Score'][maps]])
    #print(tabulate(toPrint))
    print('\nHome Team Score:',team1Score)
    print('Away Team Score:',team2Score)
    outData.to_csv("OutData.csv")

if __name__ == "__main__":
    main()