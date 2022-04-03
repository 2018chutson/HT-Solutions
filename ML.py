#hi
from pyexpat import features
import pandas as pd
import os.path
import numpy as np
import json
import geojson
import seaborn as sns
from geojson import Feature, FeatureCollection, Point
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


#import torch as tr
scaler = MinMaxScaler()
#Preps data to be worked with by the ML
def analyze(input): 
  data = pd.read_csv(input)

  data['Date'] = pd.to_datetime(data['Date'])
  data.insert(4,"Day",data['Date'].dt.day)
  data.insert(5,"Month",data['Date'].dt.month)
  data.insert(6,"Year",data['Date'].dt.year)
  DOw = data['Date'].dt.dayofweek
  DOy = data['Date'].dt.dayofyear
  wom = (data['Date'].dt.day - data['Date'].dt.weekday - 2)
  #print(DOW)
  print(data)
  print(data.columns)
  data.insert(10,"DOW",DOw)
  data.insert(11,"DOY",DOy)
  data.insert(12,"WOM",wom)

  data = data.drop('Date', axis=1)
  for i in range (0,data.shape[0]):
      time=data.iloc[i,7]
      #print(time)
      if (time[-2:] == "am"):
        data.iloc[i,7] = time.split(":")[0]
      elif (time[-2:] == "pm"):
        data.iloc[i,7] =  str(int(time.split(":")[0])+12)

  data.loc[data["Crime"] == "Abduction", "Crime"] = 1
  data.loc[data["Crime"] == "Sexual Assault", "Crime"] = 2
  data.loc[data["Crime"] == "Violation of Protection Order", "Crime"] = 3
  data.loc[data["Crime"] == "Intimidation", "Crime"] = 4
  data.loc[data["Crime"] == "Abduction", "Crime"] = 5
  data.loc[data["Crime"] == "Assault", "Crime"] = 6
  data[["Day", "Year","Time","Month","Crime"]] = data[["Day", "Year","Time","Month","Crime"]].apply(pd.to_numeric)
  
  plt.figure(figsize=(12,10))
  cor = data.corr()
  sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
  plt.show()

  months = np.arange(1,13,1)
  monthv = np.zeros(12)
  for i in range (0,data.shape[0]-1):
    monthv[data.iloc[i,5]-1] += data.iloc[i,9]
  #plt.scatter(months,monthv, color='m', label='HousVacant')
  #plt.show()

  days = np.arange(1,32,1)
  dayv = np.zeros(31)
  for i in range (0,data.shape[0]-1):
    dayv[data.iloc[i,4]-1] += data.iloc[i,9]
  #plt.scatter(days,dayv, color='m', label='HousVacant')
  #plt.show()

  years = np.arange(2016,2023,1)
  yearv = np.zeros(7)
  for i in range (0,data.shape[0]-1):
    yearv[data.iloc[i,6]-2016] += data.iloc[i,9]
  #plt.scatter(years,yearv, color='m', label='HousVacant')
  #plt.show()

  times = np.arange(1,25,1)
  timev = np.zeros(24)
  for i in range (0,data.shape[0]-1):
    timev[data.iloc[i,7]-1] += data.iloc[i,9]
  plt.scatter(times,timev, color='m', label='HousVacant')
  plt.show()
  tms=pd.DataFrame(times, index=None, columns=["time"])
  tms.insert(1,"CNT",timev)
  plt.figure(figsize=(12,10))
  cor = tms.corr()
  sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
  plt.show()

  Dows = np.arange(0,7,1)
  DOWv = np.zeros(7)
  for i in range (0,data.shape[0]-1):
    DOWv[data.iloc[i,10]] += data.iloc[i,9]
  #plt.scatter(Dows,DOWv, color='m', label='HousVacant')
  #plt.show()

  Doys = np.arange(1,366,1)
  DOyv = np.zeros(365)
  for i in range (0,data.shape[0]-1):
    DOyv[data.iloc[i,11]-1] += data.iloc[i,9]
  #plt.scatter(Doys,DOyv, color='m', label='HousVacant')
  #plt.show()
  return
####################################################################################################################################
def PrepData(input):
  data = pd.read_csv(input) 
  #splits the Date coll into day month and year 
  data['Date'] = pd.to_datetime(data['Date'])
  #print(date)
  data.insert(4,"Day",data['Date'].dt.day)
  data.insert(5,"Month",data['Date'].dt.month)
  data.insert(6,"Year",data['Date'].dt.year)
  data = data.drop('Date', axis=1)
  print(data)
  #converts time data into 24 hour single values
  for i in range (0,data.shape[0]):
      time=data.iloc[i,7]
      #print(time)
      if (time[-2:] == "am"):
        data.iloc[i,7] = time.split(":")[0]
      elif (time[-2:] == "pm"):
        data.iloc[i,7] =  str(int(time.split(":")[0])+12)
  #data['Occurences']= np.full(data.shape[0],2)
  # converts month from string to int value
  #makes sure these values are numeric 
  
  data[["Day", "Year","Time","Month"]] = data[["Day", "Year","Time","Month"]].apply(pd.to_numeric)
  print(data)


  # randomizes the data 
  #data = data.sample(frac=1, random_state=1)     
  #print(data)

  #splits data into x(features) and y 
  x = data.iloc[:,:9]
  x = x.drop('State', axis=1)
  x = x.drop('County', axis=1)
  #x = x.drop('Crime', axis=1)
  #further splits data into training and testing data
  trainx,testx,trainy,testy= train_test_split(x, data['Occurences'],
                 test_size=0.2, random_state=1, shuffle=False)
  print(trainx,trainy)
  return trainx,trainy,testx,testy

def forcastingprep(x,y):
  if (os.path.isfile('test.csv')== False):
    x.insert(6,"Occurences",y)
    x['Year'] = x['Year'].apply(str)
    x['Month'] = x['Month'].apply(str)
    x['Day'] = x['Day'].apply(str)
    print(x.dtypes)
    print(x['Year'])
    date = pd.DataFrame( x['Day'] + "/" + x['Month']+"/"+x['Year'] + "" ,columns=['D'])
    date =date.values
    #date = date.reset_index()
    #date = date.drop("index",axis=1)
    print(date)
    print(x)
    x.insert(2,"Date",date)
    print(x)
    x['Date'] = pd.to_datetime(x['Date'])
    print(x)
    temp = pd.read_csv("PredictionTemplate.csv")
    #x.to_csv('test2.csv',index=False)
    if (os.path.isfile('forcasttemplate.csv')== False):
      r = pd.date_range(start=x['Date'].min(), end=x['Date'].max())
      df = pd.DataFrame(r, columns=['Date'])
      print(df)
      #df.insert(1,"Time",times)
      len = df.shape[0]
      Dt = pd.DataFrame(columns=['Date','Time'])
      times = np.arange(1, 25, 1, dtype= int)
      for i in range (0,len):
        CurrentD= np.full(24,df.iloc[i,0])
        new = pd.DataFrame(CurrentD,columns=['Date'])
        new.insert(1,'Time',times)
        Dt = Dt.append(new,ignore_index=True)
      df = Dt
      print(df)
      final = pd.DataFrame(columns=['Latitude','Longitude','Date','Time'])
      templen = temp.shape[0]
      for i in range(0,templen):
        new = df.copy()
        lat= np.full(df.shape[0],temp.iloc[i,2])
        long= np.full(df.shape[0],temp.iloc[i,3])
        new.insert(0,"Latitude",lat)
        new.insert(1,"Longitude",long)
        final = final.append(new,ignore_index=True)
      
      final.insert(4,'Occurences',np.zeros(final.shape[0],dtype=int))
      final.to_csv('forcasttemplate.csv',index=False)
    else:
      final = pd.read_csv('forcasttemplate.csv')
    final['Date'] = pd.to_datetime(final['Date']) 
    print(x.dtypes)
    print(final.dtypes)
    print(x)
    x = x.drop('Day', axis=1)
    x = x.drop('Month', axis=1)
    x = x.drop('Year', axis=1)
    for i in range(0,x.shape[0]):
      final.loc[(final['Latitude'] == x.iloc[i,0]) & (final['Longitude'] == x.iloc[i,1]) & (final['Date'] == x.iloc[i,2]) & (final['Time'] == x.iloc[i,3]), 'Occurences'] = x.iloc[i,4]
    print(final)
    print(final['Occurences'].max())
    final.to_csv('test.csv',index=False)
    #print(final.where(final['Occurences'] > 0))  

  final = pd.read_csv('test.csv')
  final['Date'] = final['Date'].apply(str)


  date = pd.DataFrame(final['Date'].str.split("-").tolist(), columns=['year', 'month', 'day']) 
  date = date.values
  final.insert(2,"Day",date[:,2])
  final.insert(3,"Month",date[:,1])
  final.insert(4,"Year",date[:,0])
  final = final.drop('Date', axis=1)
  final[["Day", "Year","Time","Month"]] = final[["Day", "Year","Time","Month"]].apply(pd.to_numeric)
  print(final.shape[0]+1)
  xaxis = np.arange(1, final.shape[0]+1, 1, dtype= int)
  plt.figure(figsize=(12,10))
  cor = final.corr()
  sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
  plt.show()
  x = final.iloc[:,:6]
  y= final.iloc[:,6]
  return x,y


#create a ML model using lasso regression 
def Learn(x,y):
  print(x)
  print(y)
  
  #scaler.fit(x)
  #print(scaler)
  #x = scaler.transform(x)
  #x = x.iloc[:,5]
  x=x.values
  y=y.values
  #x=x.reshape(-1,1)
  print(x)
  #print(y)
  
  #transforms the features into ones with multiple degrees
  polynomial = PolynomialFeatures(degree=1, include_bias=False)
  polyx = polynomial.fit_transform(x)
  scaler = preprocessing.StandardScaler().fit(polyx)
  polyx=scaler.transform(polyx)
  print(polyx)
  # creates a model using lasso Regression
  #reg=LogisticRegression(random_state=0)
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


def forcasting(x,y):
  x=x.values
  y=y.values
  polynomial = PolynomialFeatures(degree=3, include_bias=False)
  polytestx = polynomial.fit_transform(x)
  random_forest = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=1)
  model = random_forest.fit(polytestx, y)
  predictions = model.predict(polytestx)
  r2 = r2_score(y, predictions)
  rmse = mean_squared_error(y, predictions, squared=False)
  print('The r2 is: ', r2)
  print('The rmse is: ', rmse)
  print("coef are ",model.feature_importances_)
  return(model)

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
  df = pd.read_csv("PredictionTemplate.csv")
  unique = df.shape[0]
  print(unique)
  print(df.iloc[1,:])
  temp = np.zeros(8)
  for i in range (0,unique):
    for a in range(1,13):
      df= df.append(df.iloc[i,:],ignore_index=True)
      #df=pd.concat([df,df.iloc[i,:]])
      df.iloc[df.shape[0]-1,5] = a    
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

def prediction(model,st='2022-1-1', en='2022-12-31'):
  temp = pd.read_csv("PredictionTemplate.csv")
  Drange = pd.date_range(start=st, end=en)
  r = pd.DataFrame(Drange, columns=['Date'])
  Dt = pd.DataFrame(columns=['Date','Time'])
  times = np.arange(1, 25, 1, dtype= int)  
  for i in range (0,r.shape[0]):
    CurrentD= np.full(24,r.iloc[i,0])
    new = pd.DataFrame(CurrentD,columns=['Date'])
    new.insert(1,'Time',times)
    Dt = Dt.append(new,ignore_index=True)
  df=Dt
  final = pd.DataFrame(columns=['State','County','Latitude','Longitude','Date','Time'])
  for i in range(0,temp.shape[0]):
    new = df.copy()
    lat= np.full(df.shape[0],temp.iloc[i,2])
    long= np.full(df.shape[0],temp.iloc[i,3])
    state = np.full(df.shape[0],temp.iloc[i,0])
    county = np.full(df.shape[0],temp.iloc[i,1])
    new.insert(0,"State",state)
    new.insert(1,"County",county)    
    new.insert(2,"Latitude",lat)
    new.insert(3,"Longitude",long)
    final = final.append(new,ignore_index=True)
  df= final
  df['Date'] = df['Date'].astype(str)
  print(df)
  date = pd.DataFrame(df['Date'].str.split("-").tolist(), columns=['year', 'month', 'day']) 
  date = date.values
  df.insert(4,"Day",date[:,2])
  df.insert(5,"Month",date[:,1])
  df.insert(6,"Year",date[:,0])
  df = df.drop('Date', axis=1)  
  print(df)

  x = df.iloc[:,:]
  x = x.drop('State', axis=1)
  x = x.drop('County', axis=1)
  scaled = scaler.fit_transform(x)
  x = pd.DataFrame(scaled,columns= x.columns)
  x = x.values
  polynomial = PolynomialFeatures(degree=3, include_bias=False)
  polytestx = polynomial.fit_transform(x)
  predictions = np.sum(np.multiply(polytestx,model.feature_importances_),axis=0)
  print(predictions)
  #predictions = model.predict_proba(polytestx)
  df = pd.concat([df, pd.DataFrame(predictions,columns=['predict'])],axis = 1)
  df.to_csv('prediction.csv',index=False)
 

  features = df.apply(lambda row: Feature(geometry=Point((float(row['Longitude']), float(row['Latitude'])))),axis=1).tolist()
  # all the other columns used as properties
  properties = df.drop(['Latitude', 'Longitude'], axis=1).to_dict('records')
  # whole geojson object
  feature_collection = FeatureCollection(features=features, properties=properties)
  with open('predictions.geojson', 'w', encoding='utf-8') as f: json.dump(feature_collection, f, ensure_ascii=False)

  print(df)
  df = df.sort_values('Month')
  plt.scatter(df['Month'], df['predict'], color='m', label='HousVacant')
  plt.show()
  plt.scatter(df['Day'], df['predict'], color='m', label='HousVacant')
  plt.show()
  plt.scatter(df['Time'], df['predict'], color='m', label='HousVacant')
  plt.show()
  plt.scatter(df['Latitude'], df['predict'], color='m', label='HousVacant')
  plt.show()
  plt.scatter(df['Longitude'], df['predict'], color='m', label='HousVacant')
  plt.show()

#analyze("halfCrime.csv")
x,y,testx,testy = PrepData("halfCrime.csv")
x,y = forcastingprep(x,y)
#model = Learn(x,y)
model = forcasting(x,y)
#test(model,testx,testy)
#HTpredictions(model)
prediction(model)

