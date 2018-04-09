#program for generating groups....NOT FINAL

import numpy as np
import pandas as pd
import math 
ratings = pd.read_csv('ratings1.csv')

#rm[a][b] is rating matrix in which a->movie_id & b->user_id
rm =np.ndarray(shape=(100000,944), dtype=float)
d  =np.ndarray(shape=(945,945),dtype=float)
data = np.genfromtxt('ratings1.csv',delimiter=',')

G = np.ndarray(shape=(100,944), dtype=float)
for u in range (1,944):
    for v in range (1,944):
        if u!=v:
           d1=0
           for i in range (1,1682):
               a = rm[i][u]-rm[i][v]
               b = a*a
               d1 = d1+b
           d[u][v]=math.sqrt(d1)
for G1 in range (1,944):
    min = d[d>0].min()
    G2=0
    for u in range(1,944):
        for v in range (1,944):
            if d[u][v]==min:
                G[G1][G2]=u
                G2 = G2+1
                G[G1][G2]=v
                G2 = G2+1
                d[u][v]=0
                flag=0
                for p in range(1,944):
                    for q in range(1,944):
                        if d[p][q] != 0:
                            flag = 1
                if flag ==0:
                    break
for i in range (1,944):
    print(G[1][i])                
                

           
                 