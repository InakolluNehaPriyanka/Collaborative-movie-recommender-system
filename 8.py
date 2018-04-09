#prog for group recommendation..Algo2 ....NOT FINAL

import numpy as np
import pandas as pd
#import math

ratings = pd.read_csv('ratings1.csv')
similarity = pd.read_csv('similarity1.csv')
group = pd.read_csv('group1.csv')

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

grp = np.ndarray(shape=(6,944),dtype=float)
data2 = np.genfromtxt('group1.csv',delimiter = ',')
print("group matrix starts here")
for n2 in range(1,1000):
    x2=data2[n2][0]
    y2=data2[n2][1]
    z2=data2[n2][2]
    s1=int(x2)
    t1=int(y2)
    p1=int(z2)
    
#s1 ->group_id;t1->user_id;p1 is merely a counter
    grp[s1][p1]=t1
   
print("groupmatrix is now ready")

g1=input("Enter group_id:")
c=0
user=np.ndarray(shape=(944),dtype = float)
Q = np.ndarray(shape=(944),dtype=float)
temp = np.ndarray(shape=(944),dtype=float)
temp[c] = grp[g1][c]
while (temp[c] != 0):
    
    user[c]=grp[g1][c]
    #code for caln of Q[u]
    for i in range (1,1682):
        for j in range(1,1682):
         
            nr=0
            dr=0
            max=0
            j1 = int(j)
            #j->movie_id,u->user_id
            u = int(user[c])
            if (rm[j][u])!= 0:
                new_nr = sm[i][j] * rm[j][u]
                nr = nr + new_nr
                dr = dr + sm[i][j]
        fun=nr/dr
        if max < fun:
            max = fun
            max_movId = i
            Q[u]=max_movId  
c=c+1

print("THE RECOMMENDED MOVIE LIST IS : ")
fin=np.argsort(Q)[-10:] 
print(fin)