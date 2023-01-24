import numpy as np
import pandas as pd
 
# creating a dataframe
df = pd.DataFrame({'Name': ['Raj', 'Akhil', 'Salim', 'Sonum', 'Sahil', 'Divya', 'Megha'],
                   'Age': [20, 22, 21, 19, 17, 23, 21],
                   'Rank': [1, 1, 8, 9, 4, 3, 2]})
df.sort_values(by=['Age','Rank'], ascending=[False, False], inplace=True)
print(df)