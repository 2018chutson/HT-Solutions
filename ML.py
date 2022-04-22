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
from sklearn import metrics
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
from sympy import degree


#import torch as tr
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
  plt.scatter(months,monthv, color='m', label='HousVacant')
  plt.show()

  days = np.arange(1,32,1)
  dayv = np.zeros(31)
  for i in range (0,data.shape[0]-1):
    dayv[data.iloc[i,4]-1] += data.iloc[i,9]
  plt.scatter(days,dayv, color='m', label='HousVacant')
  plt.show()

  years = np.arange(2016,2023,1)
  yearv = np.zeros(7)
  for i in range (0,data.shape[0]-1):
    yearv[data.iloc[i,6]-2016] += data.iloc[i,9]
  plt.scatter(years,yearv, color='m', label='HousVacant')
  plt.show()

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
  plt.scatter(Dows,DOWv, color='m', label='HousVacant')
  plt.show()

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
  print(data.iloc[8,7])
  time=data.iloc[8,7]
  print(time[-2:])
  #converts time data into 24 hour single values
  for i in range (0,data.shape[0]):
      time=data.iloc[i,7]
      #print(time)
      if (time[-2:] == "am"):
        data.iloc[i,7] = time.split(":")[0]
      elif (time[-7:] == "12:59pm"):
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
  x = x.drop('Crime', axis=1)
  #further splits data into training and testing data
  #trainx,testx,trainy,testy= train_test_split(x, data['Occurences'],
     #            test_size=0, random_state=1, shuffle=False)
  #print(trainx,trainy)
  #trainx = x 
  return x, data.iloc[:,9]#trainx,trainy,testx,testy

def forcastingprepwithouttimeloc(fx,fy, lat='41.293265', long= "-105.590699"):
  #print(fx)
  #print("ong",long)
  x=fx
  y=fy
  x.insert(6,"Occurences",y)
  x['Year'] = x['Year'].apply(str)
  x['Month'] = x['Month'].apply(str)
  x['Day'] = x['Day'].apply(str)
  #print(x.dtypes)
  #print(x['Year'])
  date = pd.DataFrame( x['Day'] + "/" + x['Month']+"/"+x['Year'] + "" ,columns=['D'])
  date =date.values
  #date = date.reset_index()
  #date = date.drop("index",axis=1)
 # print(date)
 # print(x)
  x.insert(2,"Date",date)
 # print(x)
  x['Date'] = pd.to_datetime(x['Date'])
  #print(x)
  #x.to_csv('test2.csv',index=False)
  r = pd.date_range(start=x['Date'].min(), end=x['Date'].max())
  df = pd.DataFrame(r, columns=['Date'])
  #print(df)
  final = pd.DataFrame(columns=['Latitude','Longitude','Date'])
  new = df.copy()
  lat= np.full(df.shape[0],lat)
  long= np.full(df.shape[0],long)
  new.insert(0,"Latitude",lat)
  new.insert(1,"Longitude",long)
  final = final.append(new,ignore_index=True)
  
  final.insert(3,'Occurences',np.zeros(final.shape[0],dtype=int))
  final['Date'] = pd.to_datetime(final['Date']) 
  #print(x.dtypes)
  #print(final.dtypes)
  #print(x)
  x = x.drop('Day', axis=1)
  x = x.drop('Month', axis=1)
  x = x.drop('Year', axis=1)
  for i in range(0,x.shape[0]):
    final.loc[(final['Latitude'] == x.iloc[i,0]) & (final['Longitude'] == x.iloc[i,1]) & (final['Date'] == x.iloc[i,2]), 'Occurences'] = final.iloc[i,3]+ x.iloc[i,4]
  for i in range(0,final.shape[0]):
    if(final.iloc[i,3] > 1 ):
      for a in range(0,final.iloc[i,3]):
        final= final.append(final.iloc[i,:],ignore_index=True)
        final.iloc[final.shape[0]-1,3]= 1
      final.iloc[i,3] = 1

  final.to_csv("testss.csv")
  final['Date']=final['Date'].astype('string')
  date = pd.DataFrame(final['Date'].str.split("-").tolist(), columns=['year', 'month', 'day']) 
  date = date.values
  final.insert(2,"Day",date[:,2])
  final.insert(3,"Month",date[:,1])
  final.insert(4,"Year",date[:,0])
  final = final.drop('Date', axis=1)
  final[["Day", "Year","Month"]] = final[["Day", "Year","Month"]].apply(pd.to_numeric)
  #print(final.shape[0]+1)
  x = final.iloc[:,:5]
  y= final.iloc[:,5]
  print(x)
  trainx,testx,trainy,testy= train_test_split(x, final['Occurences'],
                 test_size=.2, random_state=1, shuffle=True)
  #print(x)
  #print(y)
  return trainx,testx,trainy,testy


def forcasting(x,y):
  data = x
  #print(x)
  x=x.values
  y=y.values
  polynomial = PolynomialFeatures(degree=2, include_bias=False)
  polytestx = polynomial.fit_transform(x)
  columns= polynomial.get_feature_names((data.columns).values)

  random_forest = RandomForestClassifier(n_estimators=100,max_depth=None ,random_state=1)# max_depth=10
  model = random_forest.fit(polytestx, y)
  predictions = model.predict(polytestx)
  prob = model.predict_proba(polytestx)
  #print(predictions)
  rmse = mean_squared_error(y, predictions, squared=False)
  print("Accuracy on train:",metrics.accuracy_score(y,predictions))
  print('The rmse is: ', rmse)

  feature_imp = pd.Series(model.feature_importances_,index=columns).sort_values(ascending=False)
  #print(feature_imp)
  sns.barplot(x=feature_imp, y=feature_imp.index)
  #plt.show()

  data.insert(data.shape[1],"Prediciton",predictions)
  data.insert(data.shape[1],"Probability",prob[:,1])
  #data.to_csv('currentprediction.csv',index=False)
  #features = data.apply(lambda row: Feature(geometry=Point((float(row['Longitude']), float(row['Latitude'])))),axis=1).tolist()
  # all the other columns used as properties
  #properties = data.drop(['Latitude', 'Longitude'], axis=1).to_dict('records')
  # whole geojson object
  #feature_collection = FeatureCollection(features=features, properties=properties)
  #with open('currentpredictions.geojson', 'w', encoding='utf-8') as f: json.dump(feature_collection, f, ensure_ascii=False)


  pred = model.predict_proba(polytestx)
  data.insert(5,"predict",pred[:,1] )
  #print(data.iloc[:365,:])
  day = np.arange(0,365,1)
  #plt.scatter(day,y, color='g', label='HousVacant')
  plt.plot(day,data.iloc[:365,5], color='m', label='HousVacant')
  #plt.show()


  months = np.arange(1,13,1)
  monthv = np.zeros(12)
  for i in range (0,data.shape[0]-1):
    monthv[data.iloc[i,3]-1] += data.iloc[i,5]
  plt.scatter(months,monthv, color='m', label='HousVacant')
  #plt.show()

  days = np.arange(1,32,1)
  dayv = np.zeros(31)
  for i in range (0,data.shape[0]-1):
    dayv[data.iloc[i,2]-1] += data.iloc[i,5]
  plt.scatter(days,dayv, color='m', label='HousVacant')
  #plt.show()

  years = np.arange(2016,2023,1)
  yearv = np.zeros(7)
  for i in range (0,data.shape[0]-1):
    yearv[data.iloc[i,4]-2016] += data.iloc[i,5]
  plt.scatter(years,yearv, color='m', label='HousVacant')
  #plt.show()


  return(model)

def test(model,testx,testy):
  #testx = testx.drop('time', axis=1)
  #print(testx)
  polynomial = PolynomialFeatures(degree=2, include_bias=False)
  polytestx = polynomial.fit_transform(testx)
  predictions = model.predict(polytestx)
  print("Accuracy:",metrics.accuracy_score(testy,predictions))
  return metrics.accuracy_score(testy,predictions)


def currentprediction(model,st,en,lat,long,state,county):
  Drange = pd.date_range(start=st, end=en)
  df = pd.DataFrame(np.full(Drange.shape[0],state),columns=['State'])
  df.insert(1,"County",np.full(Drange.shape[0],county))
  df.insert(1,"Latitude",np.full(Drange.shape[0],lat))
  df.insert(1,"Longitude",np.full(Drange.shape[0],long))  
  df.insert(1,"Date",Drange) 
  df['Date'] = df['Date'].astype(str)
  #print(df)
  date = pd.DataFrame(df['Date'].str.split("-").tolist(), columns=['year', 'month', 'day']) 
  date = date.values
  df.insert(4,"Day",date[:,2])
  df.insert(5,"Month",date[:,1])
  df.insert(6,"Year",date[:,0])
  #df = df.drop('Date', axis=1)  
  #print(df)

  x = df.iloc[:,:]
  x = x.drop('State', axis=1)
  x = x.drop('County', axis=1)
  x = x.drop('Date', axis =1)
  #x = x.drop('Time', axis=1)

  x = x.values
  polynomial = PolynomialFeatures(degree=2, include_bias=False)
  polytestx = polynomial.fit_transform(x)
  pred = model.predict(polytestx)
  predictions = model.predict_proba(polytestx)
  #print(predictions)

  df = df.drop('Day', axis=1)
  df = df.drop('Month', axis=1)
  df = df.drop('Year', axis =1)


  df = pd.concat([df, pd.DataFrame(pred,columns=['prediction'])],axis = 1)
  df = pd.concat([df, pd.DataFrame(predictions[:,1],columns=['probability of event'])],axis = 1)
  #df.to_csv(""+county+'prediction.csv',index=False)
 

  #features = df.apply(lambda row: Feature(geometry=Point((float(row['Longitude']), float(row['Latitude'])))),axis=1).tolist()
  # all the other columns used as properties
 # properties = df.drop(['Latitude', 'Longitude'], axis=1).to_dict('records')
  # whole geojson object
  #feature_collection = FeatureCollection(features=features, properties=properties)
  ##with open(""+county+'predictions.geojson', 'w', encoding='utf-8') as f: json.dump(feature_collection, f, ensure_ascii=False)
  return df

def prediction(model,st='2022-1-1', en='2022-12-31'):
  
  temp = pd.read_csv("PredictionTempAlbany.csv")
  Drange = pd.date_range(start=st, end=en)
  r = pd.DataFrame(Drange, columns=['Date'])
  Dt = pd.DataFrame(columns=['Date','Time'])
  df = pd.DataFrame(Drange,columns=['Date'])
  times = np.arange(1, 25, 1, dtype= int)  
  #or i in range (0,r.shape[0]):
    #CurrentD= np.full(24,r.iloc[i,0])
    #new = pd.DataFrame(CurrentD,columns=['Date'])
    #new.insert(1,'Time',times)
    #Dt = Dt.append(new,ignore_index=True)
  #df=Dt
  final = pd.DataFrame(columns=['State','County','Latitude','Longitude','Date'])#,'Time'])
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
  #x = x.drop('Time', axis=1)

  x = x.values
  polynomial = PolynomialFeatures(degree=2, include_bias=False)
  polytestx = polynomial.fit_transform(x)
  pred = model.predict(polytestx)
  predictions = model.predict_proba(polytestx)
  print(predictions)
  df = pd.concat([df, pd.DataFrame(pred,columns=['prediction'])],axis = 1)
  df = pd.concat([df, pd.DataFrame(predictions[:,1],columns=['probability of event'])],axis = 1)
  df.to_csv('Albanyprediction.csv',index=False)
 

  features = df.apply(lambda row: Feature(geometry=Point((float(row['Longitude']), float(row['Latitude'])))),axis=1).tolist()
  # all the other columns used as properties
  properties = df.drop(['Latitude', 'Longitude'], axis=1).to_dict('records')
  # whole geojson object
  feature_collection = FeatureCollection(features=features, properties=properties)
  with open('Albanypredictions.geojson', 'w', encoding='utf-8') as f: json.dump(feature_collection, f, ensure_ascii=False)

  #print(df)
  #df = df.sort_values('Month')
  #plt.scatter(df['Month'], df['probability of event'], color='m', label='HousVacant')
  #plt.show()
  #df = df.sort_values('Day')
  #plt.scatter(df['Day'], df['probability of event'], color='m', label='HousVacant')
  #plt.show()
  #plt.scatter(df['Time'], df['predict'], color='m', label='HousVacant')
  #plt.show()
  return df


def fullprediction(x,y):
  data = pd.read_csv("PredictionTemplate.csv")
  prediction = pd.DataFrame(columns=['State','County','Latitude','Longitude','Date'])
  future = pd.DataFrame(columns=['State','County','Latitude','Longitude','Date'])
  Accuracy = 0
  for i in range (0,data.shape[0]):
    #print(x)
    trainx,testx,trainy,testy = forcastingprepwithouttimeloc(x.copy(),y.copy(),data.iloc[i,2],data.iloc[i,3])
    model= forcasting(trainx.copy(),trainy.copy())
    Accuracy= Accuracy + test(model,testx.copy(),testy.copy())
    print("Accuracy for ",data.iloc[i,0],data.iloc[i,1], ": ", Accuracy)
    date = pd.DataFrame( trainx['Day'].astype('string') + "/" + trainx['Month'].astype('string')+"/"+trainx['Year'].astype('string') + "" ,columns=['D'])
    date['D'] = pd.to_datetime(date['D'])
    prediction=prediction.append(currentprediction(model,date['D'].min(),date['D'].max(),data.iloc[i,2],data.iloc[i,3],data.iloc[i,0],data.iloc[i,1]),ignore_index=True)
    future =future.append(currentprediction(model,'2022-1-1','2022-12-31',data.iloc[i,2],data.iloc[i,3],data.iloc[i,0],data.iloc[i,1]),ignore_index=True)
    print(prediction)
    print(future)

  future.to_csv('futurepredictions.csv',index=False)
  features = future.apply(lambda row: Feature(geometry=Point((float(row['Longitude']), float(row['Latitude'])))),axis=1).tolist()
  # all the other columns used as properties
  properties = future.drop(['Latitude', 'Longitude'], axis=1).to_dict('records')
  # whole geojson object
  feature_collection = FeatureCollection(features=features, properties=properties)
  with open('futurepredictions.geojson', 'w', encoding='utf-8') as f: json.dump(feature_collection, f, ensure_ascii=False)

  prediction.to_csv('currentpredictions.csv',index=False)
  features = prediction.apply(lambda row: Feature(geometry=Point((float(row['Longitude']), float(row['Latitude'])))),axis=1).tolist()
  # all the other columns used as properties
  properties = prediction.drop(['Latitude', 'Longitude'], axis=1).to_dict('records')
  # whole geojson object
  feature_collection = FeatureCollection(features=features, properties=properties)
  with open('currentpredictions.geojson', 'w', encoding='utf-8') as f: json.dump(feature_collection, f, ensure_ascii=False)


  print("average accuracy", Accuracy/data.shape[0])
  
  return



xs,ys= PrepData("halfCrime.csv")
fullprediction(xs,ys)
#trainx,testx,trainy,testy = forcastingprepwithouttimeloc(xs,ys)
#model = forcasting(trainx,trainy)
#prediction(model)
#test(model,testx,testy)

