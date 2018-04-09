#program for distance matrix used in algo1[generating groups] for group recommendation....FINAL 

import numpy as np
import pandas as pd
import math
from contextlib import redirect_stdout

 
ratings = pd.read_csv('ratings1.csv')
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

d  =np.ndarray(shape=(945,945),dtype=float)
data = np.genfromtxt('ratings1.csv',delimiter=',')

G = np.ndarray(shape=(944,944), dtype=float)
for u in range (1,944):
    for v in range (1,944):
        if u!=v:
           d1=0
           for i in range (1,1682):
               a = rm[i][u]-rm[i][v]
               if a<0:
                   a = a*-1
               #print(a)
               b = a*a
               #print(b)
               d1 = d1+b
           #print(d1)    
           d2=math.sqrt(d1)
           d[u][v]=int(d2)
           print(d[u][v])
           
           if d[u][v]!=0:
             with open('distance.txt', 'a') as f:
                with redirect_stdout(f):
                    print("distance d[u][v] between", u, "and", v ,"is :", d[u][v])           
           
           
           