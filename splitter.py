import os
import pandas as pd
import sys

#~110KB initial file
df = pd.read_csv("abalone_dataset1_train.csv")
print(sys.getsizeof(df))

#creates a 104MB file
df_larger = pd.concat([df]*700, ignore_index=True)
print(sys.getsizeof(df_larger))

df_larger.to_csv("abalone-100mb.csv")
