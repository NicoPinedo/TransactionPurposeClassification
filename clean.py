# Load libraries
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt  # used to plot scatter matrix

print('Computing initial analysis of data...')

# Reads in data
data = pd.read_csv('data.csv')

# Computes number of unique values in each field of data
data.nunique(dropna = True)

# Computes number of null values in each field 
data.isnull().sum()

# Computes mean, standard deviation, min and max values for each continuous field 
continuous_names = ['Account Balance', 'Account Overdraft', 'Credit', 'Debit', 'User Year of Birth']
means = data.loc[:, continuous_names].mean()
stds = data.loc[:, continuous_names].std()
mins = data.loc[:, continuous_names].min()
maxes = data.loc[:, continuous_names].max()

# Computes scatter matrix of all continuous fields
scatter_matrix(data.loc[:, 'Account Balance':'User Year Of Birth'])
print('Close figure window to continue.')
plt.show()

print('Cleaning and reformatting data...')

# Removes erroneous rows with regard to User Year Of Birth
data = data[-(data['User Year Of Birth'] < 1900)]

# One hot encodes the following fields of data
data = pd.get_dummies(data, columns = ['Account Type', 'Transaction Type', 'User Gender'])

# Discretizes Purpose field
# Note: type changes are reset when writing and reading data frame as external file (remains for proof of concept)
data['Purpose'] = data['Purpose'].apply(str)

# Removes entries with no description data
data = data[-(data['Transaction ID'] == 429953538)]
data = data[-(data['Transaction ID'] == 480956662)]
data = data[-(data['Transaction ID'] == 480448266)]

# Removes non-feature or non-class fields
data = data.drop(['Transaction ID', 'DateTime', 'Partial Postcode', 'Local Authority District',
                  'User Year Of Birth'], axis = 1)

print('Writing cleansed data to file...')

# Writes cleansed data to new csv file
data.to_csv('preprocessed_data.csv')
