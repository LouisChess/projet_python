#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 20:33:54 2018

@author: matthieufuteral-peter
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import datetime
import geopy.distance
path="/Users/matthieufuteral-peter/Desktop/ENSAE/ENSAE_2A/Projet_python/taxi_fares/train.csv"
df=pd.read_csv(path,sep=",",nrows=200000)
#J'ai pris que les 200 000 premières lignes parce que ma mémoire saturait sinon

#df.columns #'key', 'fare_amount', 'pickup_datetime', 'pickup_longitude',
       #'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       #'passenger_count'
#df.describe()

#On transforme la key en date et time
date=[]
time=[]
for k in range(df.shape[0]):
    date.append(re.split(' ',df['key'][k])[0])
    time.append(re.split(' ',df['key'][k])[1])

for i in range(len(time)):
    time[i]=re.sub('\:.*',"",time[i])
    
df['date']=date
df['time']=time

#On trouve le jour de la semaine
week_day=[]
for j in range(df.shape[0]):
    dt=df['date'][j]
    year, month, day = (int(x) for x in dt.split('-'))    
    ans = datetime.date(year, month, day)
    week_day.append(ans.strftime('%A'))
df['week_day']=week_day

#Supression columns et outliers
del df['key'], df['pickup_datetime']
df=df[df["pickup_longitude"]!=0.000000]
#df["pickup_longitude"].max()=2140.60116
#df["pickup_latitude"].max()=1703.092772
#df["dropoff_longitude"].max()=40.851027 
#df["dropoff_latitude"].max()=404.61667
df=df[(df["pickup_longitude"]>=-180)]
df=df[(df["pickup_longitude"]<=180)]
df=df[(df["pickup_latitude"]>=-90)]
df=df[(df["pickup_latitude"]<=90)]
df=df[(df["dropoff_longitude"]>=-180)]
df=df[(df["dropoff_longitude"]<=180)]
df=df[(df["dropoff_latitude"]>=-90)]
df=df[(df["dropoff_latitude"]<=90)]
df.index=[k for k in range(df.shape[0])]

#Coordonnées en distance, faut installer geopy c'est pas mal
distance=[]
for k in range(df.shape[0]):
    coords_pickup = (df['pickup_latitude'][k],df['pickup_longitude'][k])
    coords_dropoff = (df['dropoff_latitude'][k],df['dropoff_longitude'][k])
    distance.append(geopy.distance.vincenty(coords_pickup,coords_dropoff).km)
    
def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

for i in range(len(distance)):
    distance[i]=float(truncate(distance[i],3))

df["distance"]=distance    
df=df[df["distance"]>0.1]
df=df[df["distance"]<=100]
df=df[df['fare_amount']>1]
df.index=[k for k in range(df.shape[0])]

#Test régression linéaire
vect=np.array([1 for k in range(df.shape[0])])
dist=np.array((df['distance']))
X=np.array([vect,dist])
X=X.transpose()
y=np.array((df['fare_amount']))
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
LR=LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y)
LR.fit(X_train,y_train)
LR.predict(X_test)
LR.coef_
score=r2_score(y_test,LR.predict(X_test))
print(score) #0.747302


#Vérification avec un graphe, il y a l'air d'avoir trop d'outliers
plt.plot(df['distance'],df['fare_amount'],'x')
    
#Croisement avec le jour de la semaine
df['week_day'].value_counts().plot.pie() #Répartition apparemment équitable
for k in df['week_day'].unique():
    print(k)
    print(df['fare_amount'][df['week_day']==k].describe())
    
    
    
    
    