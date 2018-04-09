#program for single user recommendation USING BOTH LCS AND PEARSON

import numpy as np
import pandas as pd

#import math


ratings = pd.read_csv('ratings1.csv')
#print(ratings.info)

#rm[a][b] is rating matrix in which a->movie_id & b->user_id
rm =np.ndarray(shape=(100000,944), dtype=float)
data = np.genfromtxt('ratings1.csv',delimiter=',')

print("rating matrix starts here")

for i in range(1,100000):  
    x=data[i][0]
    y=data[i][1] 
    z=data[i][2]
    
    b = int(x)
    a = int(y)
    
    rm[a][b]=z
    #print(rm[a][b])

print("rating matrix ends here")   

similarity = pd.read_csv('similarity1.csv')

sm=np.ndarray(shape=(84099,1682),dtype=float)
data1 =np.genfromtxt('similarity1.csv',delimiter = ',')
print("similarity matrix starts here")
for n in range(1,84099):
 
    x1=data1[n][0]
    y1=data1[n][1]
    z1=data1[n][2]
   
    s = int(x1)
    t = int(y1)
    
    sm[s][t]=z1
    #print(sm[s][t])
print("Similarity matrix has been read and system is ready to predict now....")

Lsimilarity = pd.read_csv('lcsSimilarity1.csv')

Lsm=np.ndarray(shape=(84099,1683),dtype=float)
Ldata1 =np.genfromtxt('lcsSimilarity1.csv',delimiter = ',')
print("LCS similarity matrix starts here")
for Ln in range(1,84099):
 
    Lx1=Ldata1[Ln][0]
    Ly1=Ldata1[Ln][1]
    Lz1=Ldata1[Ln][2]
   
    Ls = int(Lx1)
    Lt = int(Ly1)
    
    Lsm[Ls][Lt]=Lz1
    #print(sm[s][t])
print("LCS Similarity matrix has been read and system is ready to predict now....")

#pred is the prediction matrix
#944 users and 50 movies are only considered
pred =np.ndarray(shape=(51), dtype=float)
th = 0.85
#threshhold value is assumed to be 0.85

u1 = input('Enter the user id: ')

temp=np.ndarray(shape=(51),dtype=float)

for i in range(1,51):
    nr=0
    dr=0
    max=0
    for j1 in range(1,51):
        j = int(j1)
        u = int(u1)
        if (rm[j][u])!= 0:
             new_nr = sm[i][j] * rm[j][u]
             nr = nr + new_nr
             dr = dr + sm[i][j]
    if dr==0:
            pred[i]=0
    else:
            pred[i]=nr/dr
              
    #print(i)
    #print(pred[i])
print("THE RECOMMENDED MOVIE LIST BY USING COSINE SIMILARITY IS : ")
fin=np.argsort(pred)[-10:] 
#print(fin)
s=set(fin)
print(s)                   

#Lpred is the LCSprediction matrix
#944 users and 50 movies are only considered
Lpred =np.ndarray(shape=(51), dtype=float)
th = 0.85
#threshhold value is assumed to be 0.85

#u1 = input('Enter the user id: ')

Ltemp=np.ndarray(shape=(51),dtype=float)

for Li in range(1,51):
    Lnr=0
    Ldr=0
    Lmax=0
    for Lj1 in range(1,51):
        Lj = int(Lj1)
        u = int(u1)
        if (rm[Lj][u])!= 0:
             Lnew_nr = Lsm[Li][Lj] * rm[Lj][u]
             Lnr = Lnr + Lnew_nr
             Ldr = Ldr + Lsm[Li][Lj]
    if Ldr==0:
            Lpred[Li]=0
    else:
            Lpred[Li]=Lnr/Ldr
    
              
    #print(Li)
    #print(Lpred[Li])
print("THE RECOMMENDED MOVIE LIST BY USING LCS IS : ")
Lfin=np.argsort(Lpred)[-10:] 
#print(fin)
Ls=set(Lfin)
print(Ls) 
print("FINAL RECOMMENDATION LIST IS AS FOLLOWS:")
print(s.intersection(Ls))   