#%%
import pandas as pd
from dataprep.eda import *
from dataprep.datasets import *
df = load_dataset("titanic")
plot_correlation(df, "age")