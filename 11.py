import pandas as pd
import numpy as np

genres = pd.read_csv('genres2.csv')




data = np.genfromtxt('genres2.csv',delimiter=',')

 
print(data[5])