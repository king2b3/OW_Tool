# OW_Tool
### Map results
InData.csv holds all the map results for the season across all three of our tiers. 
  InDataHarmony and InDataDiscord are the original data for each individual tier. I wanted to try the predictor on trainning on each tier individually and separately. Didn't find much of a difference

main.py reads in this data into a pandas data frame and saves the data into OutData.csv.

This data can be viewed by opening the CSV or bu running printData.py. In this file you need to specify which teams you would like to compare, and their map stats from the season will be outputed to output.pdf

### Predictor
This process needs cleaned up, its not as dynamic as I would like. Also the input space was not really tested beyond its usage. I was able to test near an 80% accuracy, which isn't as high as I would like, but that might just be the limit of the space.
The input space is 8 values, the mode win rate for [control, assualt, hybrid, escort], and then label is a 0 if team 0 wins, 1 if type 1 wins. The network in this is a MLP feed forward network without any input bias and 50 hidden neurons. 
The network requires some work to set up as follows.

startTrain should contain the match results for whatever you would like in the trainning space. 

startTest should then contain the matches you want to inference on.

genSet.py will create the necesarry trainning files for the network to run.

genSet 

### Wants
Dyanmic way to input map results from Tranq weekly match sheets. Right now its all mannual
Different network types and params
Maybe include new Log to Inspector workshop code integration?
