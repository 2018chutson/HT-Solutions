#hi
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
#import torch as tr
def PrepData(input):
  data = pd.read_csv(input)
  #data.loc[data["County"] == "Albany", "County"] = 0
  #data.loc[data["County"] == "Campbell", "County"] = 1
  #data.Date.str.split("-").tolist()
  date = pd.DataFrame(data['Date'].str.split("-").tolist(), columns=['day', 'month', 'year']) 
  date = date.values
  print(date)

  data.insert(4,"Day",date[:,0])
  data.insert(5,"Month",date[:,1])
  data.insert(6,"Year",date[:,2])
  data = data.drop('Date', axis=1)
  #print(data)

  for i in range (0,data.shape[0]):
      time=data.iloc[i,7]
      #print(time)
      if (time[-2:] == "am"):
        data.iloc[i,7] = time.split(":")[0]
      elif (time[-2:] == "pm"):
        data.iloc[i,7] =  str(int(time.split(":")[0])+12)

  data.loc[data["Month"] == "Jan", "Month"] = 0
  data.loc[data["Month"] == "Feb", "Month"] = 1
  data.loc[data["Month"] == "Mar", "Month"] = 2
  data.loc[data["Month"] == "Apr", "Month"] = 3
  data.loc[data["Month"] == "May", "Month"] = 4
  data.loc[data["Month"] == "Jun", "Month"] = 5
  data.loc[data["Month"] == "Jul", "Month"] = 6
  data.loc[data["Month"] == "Aug", "Month"] = 7
  data.loc[data["Month"] == "Sep", "Month"] = 8
  data.loc[data["Month"] == "Oct", "Month"] = 9
  data.loc[data["Month"] == "Nov", "Month"] = 10
  data.loc[data["Month"] == "Dec", "Month"] = 11

  data[["Day", "Year","Time"]] = data[["Day", "Year","Time"]].apply(pd.to_numeric)
  data = data.sample(frac=1, random_state=1)     
  print(data)
  x = data.iloc[:,:9]
  x = x.drop('State', axis=1)
  x = x.drop('County', axis=1)
  x = x.drop('Crime', axis=1)
  y= data.iloc[:,9]
  print(x)
  print(y)
  trainx = x.iloc[:1400,:]
  trainy = y.iloc[:1400]
  testx = x.iloc[1400:,:]
  testy = y.iloc[1400:] 
  return trainx,trainy,testx,testy

def Learn(x,y):
  x=x.values
  y=y.values
  print(x)
  print(y)
  polynomial = PolynomialFeatures(degree=3, include_bias=False)
  polyx = polynomial.fit_transform(x)
  reg = linear_model.Lasso()
  #reg = LinearRegression()
  model = reg.fit(polyx, y)
  predictions = model.predict(polyx)
  r2 = r2_score(y, predictions)
  rmse = mean_squared_error(y, predictions, squared=False)
  print('The r2 is: ', r2)
  print('The rmse is: ', rmse)
  print("coef are ",model.coef_)
  print("intercept is " , model.intercept_)
  #print(model)
  return model

def test(model,testx,testy):
  polynomial = PolynomialFeatures(degree=3, include_bias=False)
  polytestx = polynomial.fit_transform(testx)
  predictions = model.predict(polytestx)
  r2 = r2_score(testy, predictions)
  rmse = mean_squared_error(testy, predictions, squared=False)
  print('The r2 is: ', r2)
  print('The rmse is: ', rmse)
  return

def HTpredictions(model):
  #df = {'Latitude':[], 
  #      'Longitude':[], 
  #      'Day':[],
  #      'Month':[],
  #      'Year':[],
  #      'Time':[]}
  df = pd.read_csv("PredictionTemplate.csv")
  unique = df.shape[0]
  print(unique)
  print(df.iloc[1,:])
  temp = np.zeros(8)
  for i in range (0,unique):
    for a in range(1,12):
      df= df.append(df.iloc[i,:],ignore_index=True)
      #df=pd.concat([df,df.iloc[i,:]])
      df.iloc[i,5] = a    
  print(df)
      
  x = df.iloc[:,:9]
  x = x.drop('State', axis=1)
  x = x.drop('County', axis=1)
  x = x.values
  polynomial = PolynomialFeatures(degree=3, include_bias=False)
  polytestx = polynomial.fit_transform(x)
  predictions = model.predict(polytestx)
  df = pd.concat([df, pd.DataFrame(predictions,columns=['predict'])],axis = 1)
  df.to_csv('prediction.csv',index=False)
  print(df)
  df = df.sort_values('Month')
  plt.scatter(df['Month'], df['predict'], color='m', label='HousVacant')
  plt.show()
  #predictions = model.predict(polytestx)

x,y,testx,testy = PrepData("halfCrime.csv")
model = Learn(x,y)
test(model,testx,testy)
HTpredictions(model)
