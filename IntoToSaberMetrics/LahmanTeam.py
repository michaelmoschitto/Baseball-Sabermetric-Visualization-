'''
Created on Jun 18, 2019

@author: MichaelMoschitto
'''

import pandas as pd
import sqlite3 as sql
from pandas._libs.skiplist import NIL
import unittest
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
# Import `mean_absolute_error` from `sklearn.metrics`
from sklearn.metrics import mean_absolute_error

class LahmanTeam():
    
    COLS = ['yearID', 'lgID', 'teamID', 'franchID', 'divID', 'Rank', 'G', 'Ghome', 'W', 'L', 'DivWin', 'WCWin', 'LgWin', 'WSWin', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'HBP', 'SF', 'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP', 'FP', 'name', 'park', 'attendance', 'BPF', 'PPF', 'teamIDBR', 'teamIDlahman45', 'teamIDretro', 'franchID', 'franchName', 'active', 'NAassoc']
    mlb_runs_per_game = {}
    labels = ''
    '''
    classdocs
    '''
    
#   

# def connect():
#     conn = sql.connect("lahman2016.sqlite")    

    def query150Games(self):
        conn = sql.connect("lahman2016.sqlite")  
        
        query = '''select * from Teams 
    inner join TeamsFranchises
    on Teams.franchID == TeamsFranchises.franchID
    where Teams.G >= 150 and TeamsFranchises.active == 'Y';
    '''
        Teams = conn.execute(query).fetchall()
        
        return Teams

# create the data frame for Query 150

    def createDF150(self, Teams):
        teamsDF = pd.DataFrame(Teams)
        
        # giving the columns names so that you can understand the data
        teamsDF.columns = self.COLS
        
#         dropping the HBP and CS columns as they are not usefull and the null messes up the data
        teamsDF.drop(['CS', 'HBP'], axis=1)
        
#         taking care of some of the other zeros in the data like SO and DP by filling them with the median
        teamsDF['SO'].fillna(teamsDF['SO'].median())  # good example of some of the Data Frame Functions and how to use them 
        teamsDF['DP'].fillna(teamsDF['DP'].median())
        
        return teamsDF

#         this is an example of how to get the data from the rows of the data frame
#             you use the iloc method and the first indices are which rows and the second are which columns from the rows
    def printEx(self, dataFrame):
        return(dataFrame.iloc[1000:1001, 0:21])
    
    def showWins(self, df):
#         %matplotlib inline

        # Plotting distribution of wins
        plt.hist(df['W'])  # I also did a scatter plot which again was just .scatter(x,y)
        plt.xlabel('Wins')
        plt.ylabel('Quantity of Teams')
        plt.title('Wins by Team')
        
        plt.show()
        
    def showWinBins(self, df):
        
#         # taking the original data frame and creating win bins collumn in it
#         df['win_bins'] = df['W'].apply(s.create_win_bins)
        
        # taking this out will give seasons before 1900 in which there were very few teams and a bad sample size
        # this is an example of cleaning the data and removing outliers that will hurt the model
        df = df[df['yearID'] > 1900]   
         
        plt.scatter(df['yearID'], df['W'], c=df['win_bins'])
        plt.xlabel('Year')
        plt.ylabel('Wins')
        plt.title('Wins Scatter Plot')
        
        plt.show()
        
    def showRunsPerYear(self, runsPerYear):
        x = []
        y = []
        
        runsPerYear = sorted(runsPerYear.items())
        
        for key in runsPerYear:
            x.append(key[0])
            y.append(key[1])
        
        print(x)
        print(y)
        
        plt.plot(x,y)
        plt.xlabel('year')
        plt.ylabel('Runs Per Year')
        
        plt.show()
        
    def ShowRperGame(self, df):
        plt.scatter(df['RperGame'], df['W'], c = 'blue')
        plt.xlabel('Runs Scored Per Game')
        plt.ylabel('Wins')
        plt.title('Runs Scored per Game vs. Wins')
        plt.show()
        
    def ShowRAperGame(self, df):
        plt.scatter(df['RAperGame'], df['W'], c = 'red')
        plt.xlabel('Runs Allowed Per Game')
        plt.ylabel('Wins')
        plt.title('Runs Allowed per Game vs. Wins')
        plt.show()
        
    def showKmeans(self, da):
        # Create K-means model and determine euclidian distances for each data point
        kmeansModel = KMeans(n_clusters=6, random_state=1)
        distances = kmeansModel.fit_transform(da)
        
        # Create scatter plot using labels from K-means model as color
        labels = kmeansModel.labels_
        self.labels = kmeansModel.labels_
        
        plt.scatter(distances[:,0], distances[:,1], c=labels)
        plt.title('Kmeans Clusters')
        
        plt.show()
        
     
#     creating bins for the wins collumn these are ways to sort the data for modeling 
    def create_win_bins(self, w):   
        if w < 50:
            return 1
        if w >= 50 and w <= 69:
            return 2
        if w >= 70 and w <= 89:
            return 3
        if w >= 90 and w <= 109:
            return 4
        if w >= 100:
            return 5
        
    def findRunsPerGame(self, df):
        
#         have to clean this of data before 1900 as well, would have done that originally but I did not want to alter previous graph looks
        
        
        runsPerYear = {}
        gamesPerYear = {}
        
        for i, row in df.iterrows():
            
            year = row['yearID']
            runs = row['R']
            games = row['G']
            
            if year in runsPerYear:
                
                runsPerYear[year] += runs
                gamesPerYear[year] += games
                
            else:

                runsPerYear[year] = runs
                gamesPerYear[year] = games
                
#         print(runsPerYear)
#         print(gamesPerYear)
         
        mlbRunsPerGame = {}
               
        for y, g in gamesPerYear.items():
            year = y
            games = g
            runs = runsPerYear[year]
            mlbRunsPerGame[year] = runs / games
            
#         print(mlbRunsPerYear)

        self.mlb_runs_per_game = mlbRunsPerGame 
        return mlbRunsPerGame
                                            #the reason for the two lines above and below this is becuase when the fucntion is passed 
                                            #to the assign function it can only take on variable and so I can't just call findRunsperGame
    def getMlbRunsPerGame(self, year):
        return self.mlb_runs_per_game[year]
    
    def assignLabel(self, year):
        if year < 1920:
            return 1
        elif year >= 1920 and year <= 1941:
            return 2
        elif year >= 1942 and year <= 1945:
            return 3
        elif year >= 1946 and year <= 1962:
            return 4
        elif year >= 1963 and year <= 1976:
            return 5
        elif year >= 1977 and year <= 1992:
            return 6
        elif year >= 1993 and year <= 2009:
            return 7
        elif year >= 2010:
            return 8
        
    def assignDecade(self, year):
        if year < 1920:
            return 1910
        elif year >= 1920 and year <= 1929:
             return 1920
        elif year >= 1930 and year <= 1939:
            return 1930
        elif year >= 1940 and year <= 1949:
            return 1940
        elif year >= 1950 and year <= 1959:
            return 1950
        elif year >= 1960 and year <= 1969:
            return 1960
        elif year >= 1970 and year <= 1979:
            return 1970
        elif year >= 1980 and year <= 1989:
            return 1980
        elif year >= 1990 and year <= 1999:
            return 1990
        elif year >= 2000 and year <= 2009:
            return 2000
        elif year >= 2010:
            return 2010
        
     
    def getSillhouetteScore(self, da):  
                # Create silhouette score dictionary
        # Create silhouette score dictionary
        s_score_dict = {}
        for i in range(2,11):
            km = KMeans(n_clusters=i, random_state=1)
            l = km.fit_predict(da)
            s_s = metrics.silhouette_score(da, l)
            s_score_dict[i] = [s_s]       
        
        return s_score_dict
        
        

    
s = LahmanTeam()    
df = s.createDF150(s.query150Games()) 

# creating the bins later to be used for modeling
df['win_bins'] = df['W'].apply(s.create_win_bins)
# this is the same process as the win bins, we are creating bins for the different years in baseball history
df['yearLabel'] = df['yearID'].apply(s.assignLabel)

# took this out of the find runs per game function so that key error wasn not thrown for a season before 1900
df = df[df['yearID'] > 1900]



dummy_df = pd.get_dummies(df['yearLabel'], prefix='era')
# Concatenate `df` and `dummy_df`
df = pd.concat([df, dummy_df], axis=1)

# uses whatever is on the .apply, in this case its the year which gets passed to the assignDecade func
#then it adds it to the data frame as a new collumn aka creating the bins 

df['decadeLabel'] = df['yearID'].apply(s.assignDecade)
decade_df = pd.get_dummies(df['decadeLabel'], prefix='decade')
df = pd.concat([df, decade_df], axis=1)


df = df.drop(['yearLabel','decadeLabel'], axis=1) #supposed to drop the year Id too but error

# s.findRunsPerYear(df) #this is only used for testing but I had a problem getting the dictionary out of the 
# print(s.mlb_runs_per_game)

# print(s.findRunsPerGame(df))

s.findRunsPerGame(df)
df['mlbRunsPerGame'] = df['yearID'].apply(s.getMlbRunsPerGame)

# print(df['mlbRunsPerGame'])

df['RperGame'] = df['R'] / df['G']
df['RAperGame'] = df['RA'] / df['G']

df['KperBB'] = df['SO'] / df['BB']
df['KperG'] = df['SO'] / df['G']

# print(df['SO'])
# print(df['G'])
# print(df['KperG'])

# print(df.corr()['W'].nlargest(8))
# print(df.corr()['W'].nsmallest(8))

# print(df['KperBB'])
# df = df.sort_values(by = 'W')
# print(df.tail(30))

# print(df.keys())


# in order to see these at the same time you have to create a figure and add 2 sublots
# s.ShowRperGame(df)
# s.ShowRAperGame(df)

# now I am going to use them in the showWin bins func
# you still pass in the data frame it just has a new collumn win bins
# s.showWinBins(df)

# s.findRunsPerYear(df) #creates the dictionaries that are used to find the runs per year in the entire league

# s.showWins(df) #wins histogram

# s.showRunsPerYear(s.findRunsPerYear(df)) #runs per year scatter


# ------------------start of the clustering and model code----------------


attributes = ['G','R','AB','H','2B','3B','HR','BB','SO','SB','RA','ER','ERA','CG',
'SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','era_1','era_2','era_3','era_4','era_5','era_6','era_7','era_8','decade_1910','decade_1920','decade_1930','decade_1940','decade_1950','decade_1960','decade_1970','decade_1980','decade_1990','decade_2000','decade_2010','RperGame','RAperGame','mlbRunsPerGame']


# for some reason the data was skewed possibly after taking things out and so in order to do the kmeans you have to drop all of the empty slots 
df = df.dropna()

data_attributes = df[attributes]
# print(data_attributes)

# print(df[attributes].nlargest(10, attributes))


# scoredict = s.getSillhouetteScore(data_attributes)

# print(scoredict)
s.showKmeans(data_attributes)

df['labels'] = s.labels

attributes.append('labels')

numeric_cols = ['G','R','AB','H','2B','3B','HR','BB','SO','SB','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','era_1','era_2','era_3','era_4','era_5','era_6','era_7','era_8','decade_1910','decade_1920','decade_1930','decade_1940','decade_1950','decade_1960','decade_1970','decade_1980','decade_1990','decade_2000','decade_2010','RperGame','RAperGame','mlbRunsPerGame','labels','W']
data = df[numeric_cols]
# print(data.head())

# Split data DataFrame into train and test sets
train = data.sample(frac=0.75, random_state=1)
test = data.loc[~data.index.isin(train.index)]


x_train = train[attributes]
y_train = train['W']
x_test = test[attributes]
y_test = test['W']

lr = LinearRegression(normalize=True)
lr.fit(x_train,  y_train)
predictions = lr.predict(x_test)

# print(predictions)

# Determine mean absolute error
mae = mean_absolute_error(y_test, predictions)

# Print `mae`
# print(mae)

# Create Ridge Linear Regression model, fit model, and make predictions
rrm = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0), normalize=True)
rrm.fit(x_train, y_train)
predictions_rrm = rrm.predict(x_test)

# print(predictions_rrm)

# Determine mean absolute error
mae_rrm = mean_absolute_error(y_test, predictions_rrm)
# print(mae_rrm)



if __name__ == '__main__':
    unittest.main()
    
