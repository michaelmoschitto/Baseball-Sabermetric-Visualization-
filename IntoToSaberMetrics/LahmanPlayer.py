'''
Created on Jun 27, 2019

@author: MichaelMoschitto
'''
import pandas as pd
import sqlite3 as sql
from pandas._libs.skiplist import NIL
import unittest
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import urllib.request
from decimal import Decimal
from numpy.f2py.auxfuncs import throw_error
import collections
import operator
import itertools

class LahmanPlayer():
    pdf = None
    cpdf = None
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
    
    def queryCollegePlayers(self):
        conn = sql.connect("lahman2016.sqlite")
        
        query = '''select * from CollegePlaying where CollegePlaying.yearID >= 2010'''
        
        collegePlayers = conn.execute(query).fetchall()
        
        return collegePlayers
    
    def createPlayersDF(self, cp):
        cp = pd.DataFrame(cp)
        COLS = ['playerID', 'schoolID', 'year']
        cp.columns = COLS
        self.cpdf = cp
    
    def createPeopleDF(self):
        conn = sql.connect("lahman2016.sqlite")
    
        query = '''select * from Master'''
        
        people = conn.execute(query).fetchall()
        
        peopleDF = pd.DataFrame(people)
        
        COLS = ['playerID', 'birthYear', 'birthMonth', 'birthDay', 'birthCountry', 'birthState', 'birthCity', 'deathYear', 'deathMonth', 'deathDay', 'deathCountry', 'deathState', 'deathCity', 'nameFirst', 'nameLast', 'nameGiven', 'weight', 'height', 'bats', 'throws', 'debut', 'finalGame', 'retroID', 'bbrefID']
        
        peopleDF.columns = COLS
        # changing the index of the player df to the player ID in order to use loc[playerid]
        # now you can idex based on the Id and not on the row number
        peopleDF.set_index("playerID", inplace=True)
        
        self.pdf = peopleDF

    def getUniquePlayers(self):
        playerIDList = []
       
        for i in range(cpdf.shape[0]):  # shape gives a tuple of the dimensions and you have to index to the 0th to get the row number
            playerID = self.cpdf.iloc[i]['playerID']
            
            if playerID not in playerIDList: #checking to see if it is in the list already 
                playerIDList.append(playerID) #creates a list of all player ids
                
        return playerIDList
    
    def getBBrefID(self, playerIDs): #uses the player ID's to index the people df and obtain bbref IDs
        bbrefIDList = []
        
        for id in playerIDs:
            bbrefID = self.pdf.loc[id, 'bbrefID']
            bbrefIDList.append(bbrefID)
            
        return bbrefIDList
    
    def scrapeWar(self, BBrefIDList):
#         "https://www.baseball-reference.com/players/a/"   arlinst01 ".shtml"
        playerWardict = {}
        
        for i in range(len(BBrefIDList)): #
#             print(BBrefIDList[i])
#             url = ""
            print(BBrefIDList[i])
            
            url =  "https://www.baseball-reference.com/players/" + BBrefIDList[i][0] + "/" + BBrefIDList[i] + ".shtml" #format of the baseball reference url is the first letter of the id/playerid
            page = urllib.request.urlopen(url) #beautiful soup elements necessary to get down to the WAR
            soup = BeautifulSoup(page, 'html.parser')
            divs = soup.find("div", class_="p1")
            
            war = divs.findChildren('p')[0].getText()
#             print(divs.findChildren('p')[0].getText())
            if float(war) > 0:
                playerWardict[BBrefIDList[i]] = float(war)
                
        return playerWardict
    
    def getWarPerCollege(self, playerWarDict):
        collegeWarDict = {}
        
        self.cpdf.set_index("playerID", inplace=True) #have to change the index of the college player dictionary to the id so its searchable using .loc

        
        for id in playerWarDict:
            college = self.cpdf.loc[id, 'schoolID']
            
            if not isinstance(college, str): #have to check becuase if a player has gone to multiple colleges the college variable comes back as a data frame, otherwise str
                college = college.iloc[-1]  
            try:
                collegeWarDict[college] += playerWarDict[id]
            except:
                collegeWarDict[college] = playerWarDict[id]
 
        sortedCollegeWar = sorted(collegeWarDict.items(), key=operator.itemgetter(1), reverse = True)
        
    
        sortedCollegeWarDict = collections.OrderedDict(sortedCollegeWar)
        return sortedCollegeWarDict

    def showWarPerCollege(self, collegeWarDict): 
#         firsttenWar = itertools.islice(collegeWarDict.items(), 0, 10)
#         print(collegeWarDict)
        
        colleges = collegeWarDict.keys() #this is getting the last 5 entires to limit to the largest 5 war
        war = collegeWarDict.values() #   "     " 
        colleges = list(colleges)[:5]
        war = list(war)[:5]
        
        print(colleges) 
         
        plt.bar(colleges, war)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Colleges")
        plt.ylabel("Total War 2010 - 2016")
        plt.title("Best Baseball Colleges by WAR Since 2010")
        
        plt.show()
           
p = LahmanPlayer()

# done this way to create a global variable in the class so that it was easier to access the dataframe in later methods
collegePlayers = p.queryCollegePlayers()
p.createPlayersDF(collegePlayers)

cpdf = p.cpdf
# print(cpdf)

p.createPeopleDF()
pdf = p.pdf
# print(pdf.head())

playerIDList = p.getUniquePlayers()
# print('player ids: ', playerIDList)

BBrefIDList = p.getBBrefID(playerIDList)
# print('bbref ids: ', BBrefIDList)

playerWarDict = p.scrapeWar(BBrefIDList)
 
collegeWarDict = p.getWarPerCollege(playerWarDict)

p.showWarPerCollege(collegeWarDict)
 
# print(collegeWarDict)

# cpdf.set_index("playerID", inplace=True)
# print(cpdf.loc['aardsda01', 'schoolID'].iloc[-1])


# print(cpdf.size)
# print(cpdf[cpdf.size - 10])
# print(cpdf.iloc[4]['playerID']) #iterate through the college player data frame grabbing the playerID's

# print(cpdf.tail(200))
# print(cpdf['playerID'][1])

# print(pdf.loc['aardsda01', 'bbrefID'])
