import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff

# Wczytywanie zbioru danych z pliku ARFF i konwertowanie go na DataFrame
    # Input:
    #       path - ścieżka do pliku ARFF, który ma być wczytany
    # Output:
    #       df - DataFrame zawierający dane z pliku ARFF
def load_dataset_to_dataframe(path):
    dataset = arff.loadarff(path)
    df = pd.DataFrame(dataset[0])
    return df 
