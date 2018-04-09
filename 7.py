import numpy as np
import pandas as pd
from contextlib import redirect_stdout

distance = pd.read_csv('distance1.csv')

d=np.ndarray(shape=(888306,945),dtype=float)
data2 =np.genfromtxt('distance1.csv',delimiter = ',')
print("distance matrix starts here")
for n in range(1,888306):
 
    x1=data2[n][0]
    y1=data2[n][1]
    z1=data2[n][2]
   
    s = int(x1)
    t = int(y1)
    
    d[s][t]=z1
    #print(d[s][t])
print("distance matrix has been read and system is ready to generate groups now....")
G = np.ndarray(shape=(100,945), dtype=float)
count = np.ndarray(shape=(945),dtype=float)
for c in range(1,945):
    count[c]=c
#G is a matrix that represents groups with each row representing a group
for G1 in range (1,945):
    min = d[d>0].min()
    print("Group",G1,"with a distance of",min )
    G2=0
    for u in range(1,888306):
        for v in range (1,945):
            if d[u][v]==min:
                G[G1][G2]=u
                count[u]=0
                print(G[G1][G2])
                with open('group1.txt', 'a') as f:
                    with redirect_stdout(f):
                        print("group no:", G1, "members are", G[G1][G2])  
                G2 = G2+1
                G[G1][G2]=v
                count[v]=0
                print(G[G1][G2])
                with open('group1.txt', 'a') as f:
                    with redirect_stdout(f):
                        print("group no:", G1, "members are", G[G1][G2])  
                G2 = G2+1
                d[u][v]=0
                flag=0
                for p in range(1,945):
                    
                        if count[p] != 0:
                            flag = 1
                            
                if flag ==0:
                    break
