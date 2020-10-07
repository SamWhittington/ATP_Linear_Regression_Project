import codecademylib3_seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
raw_data = pd.read_csv('tennis_stats.csv')
df = pd.DataFrame(raw_data)
print(df)


x = df[['Aces', 'DoubleFaults','FirstServe',	'FirstServePointsWon','FirstServeReturnPointsWon']]

#print(x) #Selected columns of the data frame

# y = df[['Winnings']] #selecting a depedent var
#print(y)

# perform exploratory analysis here:

plt.title('First serve Vs Wins')
plt.xlabel('First serve')
plt.ylabel('Wins')
x3 = df[['FirstServe']]
y3 = df[['Wins']]
plt.scatter(x3, y3)
plt.show() 
#Interestingly firs more of a Gaussian dist than linear.

x1 = df[['Wins']]
y1 = df[['Winnings']]
plt.scatter(x1, y1)
plt.show() # relation between Wins vs Winnings

plt.title('Wins and BreakPointsOpportunities vs Winnings')
plt.xlabel('Wins and BreakPointsOpportunities')
plt.ylabel('Winnings')
x2 = df[['BreakPointsOpportunities']]
y2 = df[['Winnings']]
plt.scatter(x2, y2)
plt.show() 





## Single feature linear regressions:




x_features = df[["BreakPointsOpportunities"]]
y_winnings = df[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
slm = LinearRegression()
slm.fit(x_train, y_train)
pred = slm.predict(x_test)
print('Prediction of Winnings with BreakPointsOpportunities Test Score:',slm.score(x_test,y_test))
plt.scatter(y_test,pred,marker = 'o', alpha = 0.3)
plt.title('Predicted Winnings vs. Actual Winnings with Break Points Won as Feature')
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()


#Another linear regression model
x_features = df[["Wins"]]
y_winnings = df[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
slm = LinearRegression()
slm.fit(x_train, y_train)
pred = slm.predict(x_test)
print('Prediction of Winnings with wins Test Score:',slm.score(x_test,y_test))
plt.scatter(y_test,pred,marker = 'o', alpha = 0.3)
plt.title('Predicted Winnings vs. Actual Winnings with Wins as Feature')
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()



#Another linear regression model
x_features = df[["Aces"]]
y_winnings = df[["Wins"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
slm = LinearRegression()
slm.fit(x_train, y_train)
pred = slm.predict(x_test)
print('Prediction of Wins with Aces Test Score:',slm.score(x_test,y_test))
plt.scatter(y_test,pred,marker = 'o', alpha = 0.3)
plt.title('Predicted Wins vs. Actual Wins with Number of Aces as Feature')
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()
# Has outliers which would need to be removed to give this a better predicted result (score is ~0.65)






## perform two feature linear regressions here:

# Two features linear regression
x_features = df[["BreakPointsOpportunities",'FirstServeReturnPointsWon']]
y_winnings = df[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
double = LinearRegression()
double.fit(x_train, y_train)
pred = double.predict(x_test)
print('Prediction of Winnings with 2 Features Test Score:',double.score(x_test,y_test))
plt.scatter(y_test,pred, alpha = 0.4)
plt.title("Predicted outcome vs Actual outcome with two features")
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()

#Another two features linear regression
x_features = df[["BreakPointsOpportunities",'ServiceGamesPlayed']]
y_winnings = df[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
double = LinearRegression()
double.fit(x_train, y_train)
pred = double.predict(x_test)
print('Prediction of Winnings with 2 Features Test Score:', double.score(x_test,y_test))
plt.scatter(y_test,pred, alpha = 0.4)
plt.title("Predicted outcome vs Actual outcome with two features")
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()



## Multiple feature linear regressions:


#I classified my model as multiple which represent multiple features

# Multiple features linear regression 1
x_features = df[["BreakPointsOpportunities",'ServiceGamesPlayed',"TotalPointsWon","TotalServicePointsWon","DoubleFaults","BreakPointsConverted",'SecondServeReturnPointsWon']]
y_winnings = df[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
multiple = LinearRegression()
multiple.fit(x_train, y_train)
pred = multiple.predict(x_test)
print('Predicting Winnings with Multiple Features Test Score:',multiple.score(x_test,y_test))
plt.scatter(y_test,pred, alpha = 0.4)
plt.title("Predicted outcome vs Actual outcome with Multiple features")
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()
plt.clf()



#Another Multiple lienar regression
x_features = df[["BreakPointsOpportunities",'ServiceGamesPlayed',"TotalPointsWon","TotalServicePointsWon","BreakPointsConverted",'SecondServeReturnPointsWon','BreakPointsConverted',"BreakPointsFaced","ReturnGamesWon","SecondServePointsWon"]]
y_winnings = df[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x_features, y_winnings, train_size = 0.8, test_size = 0.2)
multiple = LinearRegression()
multiple.fit(x_train, y_train)
pred = multiple.predict(x_test)
print('Predicting Winnings with Multiple Features Test Score:', multiple.score(x_test,y_test))
plt.title("Predicted outcome vs Actual outcome with Multiple features")
plt.scatter(y_test,pred, alpha = 0.4)
plt.xlabel("Actual Outcome")
plt.ylabel("Predicted Outcome")
plt.show()


# Some programative analysis to loop through variables and find which are the best indicators automatically

# Group assumes values are either related to player ID, Variables,or performance indicators 
PlayerID = ['Player','Year']
Variables = ['FirstServe','FirstServePointsWon',
 'FirstServeReturnPointsWon','SecondServePointsWon',
 'SecondServeReturnPointsWon','Aces','BreakPointsConverted',
 'BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved',
 'DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon',
 'ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
 'TotalServicePointsWon','Win-Loss Ratio']
PerformanceIndicators = ['Wins','Winnings','Ranking']

#  Win/Loss ratio
df['Win-Loss Ratio'] =df.apply(lambda r : float(r.Wins) / r.Losses if r.Losses > 0 else 0, axis = 1)
# View available stats
print('Available Stats:\n')
for stat in df.columns.values:
    print(stat)


# Set the threshold for correlation:
Threshold = 0.3

# lists of 'Good' Correlations for predicting performance indicators:
# N.B in theory (and by experiment) performance indictors should correlate with each other so I've manually added them in
New_vars_wins = ['Winnings','Ranking']
New_vars_winnings = ['Wins','Ranking']
New_vars_ranking = ['Wins','Winnings']

# iterate through vars to see which are good predictors of each PI:
for var in Variables:
    for Indicator in PerformanceIndicators:
        x = df[[var]]
        y = df[Indicator]

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

        # build regression model
        mlr = LinearRegression()
        mlr.fit(x_train,y_train)
        y_predict = mlr.predict(x_test)

        # plt.scatter(y_test,y_predict,alpha = 0.5)
        # plt.show()
        print('\nMODEL PERFORMACE FOR: {I} vs. {V}'.format(I = Indicator, V = var))
        print("Train score: {}". format(mlr.score(x_train, y_train)))
        print("Test score: {}". format(mlr.score(x_test, y_test)))

        if mlr.score(x_test, y_test) > Threshold:
            if Indicator == 'Wins':
                New_vars_wins.append(var)
            if Indicator == 'Winnings':
                New_vars_winnings.append(var)
            if Indicator == 'Ranking':
                New_vars_ranking.append(var)
        # Give ranking a reduced threshold
        if mlr.score(x_test, y_test) > Threshold/3 and Indicator == 'Ranking':
            New_vars_ranking.append(var)


print('\nGOOD PREDICTORS FOR WINS ARE:')
for i in New_vars_wins:
    print('-{}'.format(i))
print('\nGOOD PREDICTORS FOR WINNINGS ARE:')
for i in New_vars_winnings:
    print('-{}'.format(i))
print('\nGOOD PREDICTORS FOR RANKINGS ARE:')
for i in New_vars_ranking:
    print('-{}'.format(i))

# GOOD PREDICTORS FOR RANKINGS ARE:
# -Wins
# -Winnings
# -BreakPointsOpportunities
# -ServiceGamesPlayed


# Hence use these to check if better than the above models froma guessing

# use our good correlators to predict PIs:
for Indicator, Vars in zip(PerformanceIndicators, [New_vars_wins, New_vars_winnings,New_vars_ranking]):
    x = df[Vars]
    y = df[Indicator]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

    # build regression model
    mlr = LinearRegression()
    mlr.fit(x_train,y_train)
    y_predict = mlr.predict(x_test)

    # Plot Datapoints
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    ax = plt.subplot()
    plt.scatter(y_test,y_predict,alpha = 0.5)
    ax.set_xlabel('{} Test Values'.format(Indicator))
    ax.set_ylabel('{} Predicted Values'.format(Indicator))
    ax.set_title("Optimised Test vs. Predicted Values for {I} in Tennis\nTest score: {S}".format(I = Indicator,S = mlr.score(x_test, y_test)))
    plt.savefig("Optimised Test vs. Predicted Values for {} in Tennis.png".format(Indicator))
    plt.show()
    print('\nOPTIMISED MODEL PERFORMACE FOR: {I}'.format(I = Indicator))
    print("Train score: {}". format(mlr.score(x_train, y_train)))
    print("Test score: {}". format(mlr.score(x_test, y_test)))



# Results show that this optimised test gives better results - wins are best predicted and rankings the worst, I need to perform further analysis to improve these predictions.. Credit to  AnnonymousRacoon for the colab work.
