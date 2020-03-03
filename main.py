#   Created by Bayley King (Booth OW)
#   Python 3.7
#   Started on 12/16/2019
#   Github link 

import pandas as pd

data = pd.read_csv("InData.csv", dtype=object)

for row in data.iterrows():
    print(row)

def search(choice,name,tier):
    # 
    print("search")


class Team():

    def __init__(self, tier, name):
        print('search')


class Player():

    def __init__(self, tier, name):
        print("search")
