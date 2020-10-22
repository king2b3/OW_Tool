
'''
function to load EEG data from their respective text files
Each input in every list is a list in itself with the first list value being the 
eeg signal in discrete time, and the second value of the list being the numeric label.
ie. [[0,0,0,0,.4,.34,.23, .... ,0,0],[1]] 
'''
import csv
import pandas as pd
import pickle as pkl
trainData = []
trainLabel = []
with open('startTrain.csv') as csv_file:
    lines = csv.reader(csv_file, delimiter=',')
    next(lines)
    for r in lines:
        trainData.append([r[1],r[2]])
        #print(r)
        if int(r[5]) > int(r[6]):
            trainLabel.append([1,0])
        else:
            trainLabel.append([0,1])
    csv_file.close()

testData = []
testLabel = []
with open('startTest.csv') as csv_file:
    lines = csv.reader(csv_file, delimiter=',')
    next(lines)
    for r in lines:
        testData.append([r[1],r[2]])
        #print(r)
        if int(r[5]) > int(r[6]):
            testLabel.append([1,0])
        else:
            testLabel.append([0,1])
    csv_file.close()



outData = pd.read_csv("OutData.csv",dtype=object,index_col=0)

trainOutput = []
testOutput = []

for game in trainData:
    temp = []
    for team in game:
        for mode in ['Assault','Control','Escort','Hybrid']:
            if int(outData.loc[team,mode+' W']) + int(outData.loc[team,mode+' L']) == 0:
                temp.append(0)
            else:
                temp.append((int(outData.loc[team,mode+' W']) / (int(outData.loc[team,mode+' W']) + int(outData.loc[team,mode+' L']))))
    trainOutput.append(temp)

for game in testData:
    temp = []
    for team in game:
        for mode in ['Assault','Control','Escort','Hybrid']:
            if int(outData.loc[team,mode+' W']) + int(outData.loc[team,mode+' L']) == 0:
                temp.append(0)
            else:
                temp.append((int(outData.loc[team,mode+' W']) / (int(outData.loc[team,mode+' W']) + int(outData.loc[team,mode+' L']))))
    testOutput.append(temp)



pkl.dump(trainOutput, open("trainX.pkl", "wb" ) )
pkl.dump(trainLabel, open("trainY.pkl", "wb" ) )
pkl.dump(testOutput, open("testX.pkl", "wb" ) )
pkl.dump(testLabel, open("testY.pkl", "wb" ) )



with open("trainX.csv", 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for o in trainOutput:
        csv_writer.writerow(o)

with open("trainY.csv", 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for l in trainLabel:
        csv_writer.writerow(l)

with open("trainX.csv", 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for o in testOutput:
        csv_writer.writerow(o)

with open("trainY.csv", 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for l in testLabel:
        csv_writer.writerow(l)