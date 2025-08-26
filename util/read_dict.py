import pandas as pd
import numpy as np

# Read the entire data dictionary
dd = pd.read_excel('DS13-02 Data dictionary.xlsx')
print('Complete Data Dictionary Shape:', dd.shape)
print('Columns:', list(dd.columns))
print('\n=== FULL DATA DICTIONARY ===')
print(dd)
