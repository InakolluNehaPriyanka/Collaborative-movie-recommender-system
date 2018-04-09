#program for single user recommendation 

import numpy as np
import pandas as pd

#import math
from contextlib import redirect_stdout

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
    pred[i]=nr/dr
              
    #print(i)
    #print(pred[i])
print("THE RECOMMENDED MOVIE LIST ACC. TO SINGLE LEVEL FILTERING  IS : ")
fin=np.argsort(pred)[-10:] 
#print(fin)
s=set(fin)
print(s)                   

    