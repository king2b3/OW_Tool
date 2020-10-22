import pandas as pd
from os import system, name
from time import sleep
import pdfkit as pdf
 

def clear():
    _ = system('clear')


clear()
toPrint = []
outData = pd.read_csv("OutData.csv",dtype=object,index_col=0)

team1 = "Void"
test1 = outData.loc[team1]
test1 = test1.to_frame()


team2 = "Delta Threat"
test2 = outData.loc[team2]
test2 = test2.to_frame()

temp = pd.concat([test1,test2],axis=1,join='inner')

print(temp)


temp.to_html('test.html')

path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdf.configuration(wkhtmltopdf=path_wkhtmltopdf)


pdf.from_file('test.html', 'output.pdf',configuration=config)
#pdf.from_file('test.html', 'output.pdf')