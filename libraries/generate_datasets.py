import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Generowanie plam oraz rysowanie ich na wykresie (redukcja wymiarów przy pomocy PCA)
# Input: 
#        n - liczba całkowita punktów
#        c - wymiar danych
#        k - liczba plam
#        seed - seed jakim generujemy dane
# Output: 
#       X_train - dane treningowe
#       y_train_extended - przzyporządkowane klasy do danych treningowych
#       X_test - dane testowe
#       y_test_extended - przzyporządkowane klasy do danych testowych

def generate_dataset_blobs(n, c, k, seed):
    # Generowanie syntetycznych danych
    X_train, y_train_extended = make_blobs(n_samples=n, n_features=c, centers=k, random_state=seed)
    
    X_test, y_test_extended = make_blobs(n_samples=n, n_features=c, centers=k, random_state=100)

    if X_train.shape[1]>1:
        # Redukcja wymiarowości do 2D za pomocą PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_train)
        
        # Wizualizacja danych
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y_train_extended, palette='viridis', s=100, edgecolor='k', alpha=0.7)
        plt.title("Syntetyczne dane klastrowe zredukowane do 2D za pomocą PCA")
        plt.xlabel("Pierwsza składowa główna")
        plt.ylabel("Druga składowa główna")
        plt.legend(title="Klaster")
        plt.grid(True)
        plt.show()

    return X_train, y_train_extended, X_test, y_test_extended

# Generowanie syntetycznych etykiet na podstawie średnich wartości cech
    # Input:
    #       data - 2D tablica, gdzie wiersze to próbki, a kolumny to cechy
    #       num_classes - liczba klas do przypisania
    # Output:
    #       labels - tablica z etykietami przypisanymi do każdej próbki
def generate_synthetic_labels(data, num_classes):
    # Zakładam, że 'data' to 2D tablica, gdzie wiersze to próbki, a kolumny to cechy
    num_samples, num_features = data.shape
    
    # Oblicz etykiety dla każdego wiersza w tablicy 'data'
    labels = np.zeros(num_samples, dtype=int)
    
    # Przypisz etykiety wzdłuż osi pionowej (czyli dla każdego wiersza)
    for i in range(num_samples):
        sample = data[i, :]
        # Podziel przestrzeń cechową (dla każdego wiersza osobno) na 'num_classes' segmentów
        labels[i] = np.digitize(sample.mean(), bins=np.linspace(data.min(), data.max(), num_classes + 1)) - 1
    
    return labels

    # Generowanie danych sinusoidalnych w wielu wymiarach z dodanym szumem
    # oraz wizualizacja danych po redukcji wymiarów przy pomocy PCA
    # Input:
    #       num_dimensions - liczba wymiarów danych
    #       frequency - częstotliwość sinusoidy
    #       amplitude - amplituda sinusoidy
    #       phase - lista faz dla każdego wymiaru
    #       duration - czas trwania danych
    #       sampling_rate - częstotliwość próbkowania
    #       noise_std - odchylenie standardowe szumu gaussowskiego
    #       num_classes - liczba klas do wygenerowania etykiet
    # Output:
    #       t - wektor czasu
    #       data - 2D tablica danych sinusoidalnych z dodanym szumem
    #       labels - tablica syntetycznych etykiet dla danych
    
def generate_multidimensional_sine_data(num_dimensions, frequency, amplitude, phase, duration, sampling_rate, noise_std, num_classes):
    t = np.linspace(0, duration, duration * sampling_rate)  # wektor czasu
    data = np.zeros((len(t), num_dimensions))
    
    for i in range(num_dimensions):
        noise = np.random.normal(0, noise_std, t.shape)  # szum gaussowski
        data[:, i] = amplitude * np.sin(2 * np.pi * frequency * t + phase[i]) + noise

    if data.shape[1] > 1:
        # Przeprowadzenie PCA
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)
        
        # Wykres PCA
        plt.figure(figsize=(10, 5))
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c=t, cmap='viridis')
        plt.colorbar(label='Czas')
        plt.xlabel('Główna składowa 1')
        plt.ylabel('Główna składowa 2')
        plt.title('Wizualizacja danych przy użyciu PCA')
        plt.grid(True)
        plt.show()

    # Generowanie syntetycznych etykiet
    labels = generate_synthetic_labels(data, num_classes)

    return t, data, labels


# Funkcja do generowania danych y z szumem
def generate_line(a, b, x, std_dev):
    noise = np.random.normal(0, std_dev, x.shape)
    y = a * x + b + noise
    return y
    
def generate_dataset_lines(a_values, b_values, num_points = 1000, std_dev = 0.1, n_clusters = 4):
    # Generowanie danych
    x = np.linspace(0, 10, num_points)  # Wartości x od 0 do 10
    
    # Generowanie wielowymiarowych danych
    X_train = []
    X_test = []
    for a, b in zip(a_values, b_values):
        y_train = generate_line(a, b, x, std_dev)
        y_test = generate_line(a, b, x, std_dev)
        X_train.append(y_train)
        X_test.append(y_test)
        
    # Konwertowanie listy na tablicę numpy
    X_train = np.array(X_train).T
    X_test = np.array(X_test).T
    
    y_train = generate_synthetic_labels(X_train, n_clusters)
    y_test = generate_synthetic_labels(X_test, n_clusters)
    
    # Wizualizacja danych
    plt.figure(figsize=(10, 6))
    for i, (a, b, y) in enumerate(zip(a_values, b_values, X_train.T)):
        plt.plot(x, y, label=f'y = {a}x + {b}')
    plt.title('Wielowymiarowe dane z szumem')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    return X_train, y_train, X_test, y_test

