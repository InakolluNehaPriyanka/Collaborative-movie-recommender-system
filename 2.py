#prog for generating rate, mean and similarity matrix .... NOT FINAL

import pandas as pd
import numpy as np
import math


ratings = pd.read_csv('ratings1.csv')


#rm[a][b] is rating matrix in which a->movie_id & b->user_id
rm =np.ndarray(shape=(163950,700), dtype=float)
data = np.genfromtxt('ratings1.csv',delimiter=',')

for i in range(1,100005):  
    x=data[i][0]
    y=data[i][1] 
    z=data[i][2]
    b = int(x)
    a = int(y)
    
    rm[a][b]=data[i][2]
    #print(rm[a][b])

print("rating matrix ends here")   

mean=np.ndarray(shape=(163950),dtype=float)
print("mean matrix starts here")   

for i in range(1,163950):
    sum=0
    count=0
    for j in range(1,700):
        if rm[i][j]!=0:
            sum = sum + rm[i][j]
            count = count+1
    if count!=0:
        mean[i]=sum/count
        #print(mean[i])
    else:
        mean[i]=0
        #print(mean[i])
print("mean matrix ends here...")   
#sm[a][b] is similarity matrix in which a->movie1 & b->movie2
sm=np.ndarray(shape=(163950,163950),dtype=str)
print("similarity  matrix starts here...")

for i in range(1,163950):
   
    for j in range(1,163950):
        if i==j:
            sm[i][j]=1
            print(sm[i][j])
        else:
            nr=0
            dr1=0
            dr2=0
            for k in range(1,671):
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
            print(sm[i][j]) 
print("similarity matrix ends here...")











