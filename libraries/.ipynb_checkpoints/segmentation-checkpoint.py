import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff

# Przekształcanie szeregów czasowych w jeden DataFrame
    # Input:
    #       df - DataFrame, gdzie pierwsza kolumna zawiera dane szeregów czasowych, a druga klasyfikację
    # Output:
    #       stacked - DataFrame, gdzie każdy wiersz zawiera scalone dane z poszczególnych szeregów czasowych
    #       labels - tablica z etykietami klasyfikacyjnymi dla każdego szeregów czasowych
    
    # Założenie, że wymiar danych to liczba wierszy pierwszego elementu w DataFrame
def stack_time_series(df):

    dimensionality = df.iloc[0,0].shape[0] # assumption that data dimensionality is the number of rows of first element in dataframe
    # also the first column is expected to store data (second stores classification)
    
    # creating new data frame 
    stacked = pd.DataFrame(index=np.arange(dimensionality), columns=np.arange(1))

    labels = []
    # filling it with empty numpy arrays
    for i in range(dimensionality):
        stacked.iloc[i, 0] = np.array([])
    # filling the dataframe with data    
    for index, row in df.iterrows():
        i = 0
        labels.append(row[1])
        for r in row[0]:
             stacked.iloc[i,0] = np.concatenate((stacked.iloc[i,0],np.array(r.tolist())))
             i += 1
            
    return stacked, np.array(labels)

# Przekształcanie szeregów czasowych w jeden DataFrame z losowym przemieszaniem danych i identyfikowaniem punktów zmiany
# Input:
#       df - DataFrame, gdzie pierwsza kolumna zawiera dane szeregów czasowych, a druga klasyfikację
# Output:
#       stacked - DataFrame, gdzie każdy wiersz zawiera scalone dane z poszczególnych szeregów czasowych
#       change_points - tablica punktów zmiany w danych, wskazująca miejsca, gdzie zmienia się klasa
#       labels - tablica z etykietami klasyfikacyjnymi dla każdego szeregów czasowych
def stack_time_series_randomly(df, seed=43):

    dimensionality = df.iloc[0,0].shape[0] # assumption that data dimensionality is the number of rows of first element in dataframe
    # also the first column is expected to store data (second stores classification)

    change_points = np.array([],dtype=int)
    
    # creating new data frame 
    stacked = pd.DataFrame(index=np.arange(dimensionality), columns=np.arange(1))

    labels = []
    # randomize rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # filling it with empty numpy arrays
    for i in range(dimensionality):
        stacked.iloc[i, 0] = np.array([])

    last_class = df.iloc[0,1]

    # filling the dataframe with data    
    for index, row in df.iterrows():
        i = 0
        labels.append(row[1])
        if last_class != row[1]:
            change_points = np.append(change_points,int(index*100))
        last_class = row[1]
        for r in row[0]:
             stacked.iloc[i,0] = np.concatenate((stacked.iloc[i,0],np.array(r.tolist())))
             i += 1
        
    return stacked, change_points, labels
