#program for single user recommendation using lcs

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


#Lpred is the LCSprediction matrix
#944 users and 50 movies are only considered
Lpred =np.ndarray(shape=(51), dtype=float)
th = 0.85
#threshhold value is assumed to be 0.85

u1 = input('Enter the user id: ')

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
    Lpred[Li]=Lnr/Ldr
              
    print(Li)
    print(Lpred[Li])
print("THE RECOMMENDED MOVIE LIST IS : ")
with open('check.txt', 'a') as f:
                with redirect_stdout(f):
                       Lfin=np.argsort(Lpred)[-10:] 
                       print(Lfin)


    