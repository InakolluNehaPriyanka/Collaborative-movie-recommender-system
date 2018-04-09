#program for generating similarity matrix....FINAL

import numpy as np
import pandas as pd
import math
from contextlib import redirect_stdout

ratings = pd.read_csv('ratings1.csv')
#print(ratings.info)


#rm[a][b] is rating matrix in which a->movie_id & b->user_id
rm =np.ndarray(shape=(100001,944), dtype=float)
data = np.genfromtxt('ratings1.csv',delimiter=',')
print("rating matrix starts here")
for i in range(1,100001):  
    x=data[i][0]
    y=data[i][1] 
    z=data[i][2]
    b = int(x)
    a = int(y)
    
    rm[a][b]=z
    #print(rm[a][b])

print("rating matrix ends here")   

    
mean=np.ndarray(shape=(1682),dtype=float)
print("mean matrix starts here")   

for i in range(1,1682):
    sum=0
    count=0
    for j in range(1,944):
        if rm[i][j]!=0:
            sum = sum + rm[i][j]
            count = count+1
    if count!=0:
        mean[i]=sum/count
        #print ("mean value of movieId" ,i ,"is :")
        #print(mean[i])
    else:
        print ("mean value of movieId" ,i ,"is :")
        mean[i]=0
        print(mean[i])
print("mean matrix ends here...") 
#sm[a][b] is similarity matrix in which a->movie1 & b->movie2
g = list()
sm=np.ndarray(shape=(100001,1682),dtype=float)
print("similarity  matrix starts here...")

for i in range(1,1682):
   
    for j in range(1,1682):
        if i==j:
            sm[i][j]=1
            with open('help.txt', 'a') as f:
                with redirect_stdout(f):
                    print("similarity between", i, "and", j ,"is :", sm[i][j])
                    g.append(sm[i][j])
        else:
            nr=0
            dr1=0
            dr2=0
            for k in range(1,944):
                #nr is numerator & dr is denominator in similarity formula
                
                t1= (rm[i][k]-mean[i])
                t2= (rm[j][k]-mean[j])
                nr1=t1*t2
                dr11=t1*t1
                dr21=t2*t2
                nr=nr+nr1
                dr1=dr1+dr11
                dr2=dr2+dr21
                #print("nr")
                #print(nr)
                
            sqdr1=math.sqrt(dr1)
            sqdr2=math.sqrt(dr2)
            dr=sqdr1 * sqdr2
            #print("dr")
            #print(dr)
            if nr==0 or dr==0:
                    sm[i][j]=0
            else: 
                    sm[i][j]=nr/dr
        if sm[i][j]!=0:
             with open('help.txt', 'a') as f:
                with redirect_stdout(f):
                    print("similarity between", i, "and", j ,"is :", sm[i][j])
                    
    
print("similarity matrix ends here...")
  


