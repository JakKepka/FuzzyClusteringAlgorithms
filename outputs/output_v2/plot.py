import os
import csv
from collections import defaultdict

def save_best_params(dataset_name, algorithm_name, classifier_name, best_params, centroids, accuracy, precision, recall, f1, tuning=False):
    """
    Zapisuje najlepsze hiperparametry wraz z metrykami do pliku CSV.
    """
    filename = f"{dataset_name}_tuning.csv" if tuning else f"{dataset_name}.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            headers = ["Dataset", "Algorithm", "Classifier", "Centroids", "Accuracy", "Precision", "Recall", "F1"] + list(best_params.keys())
            writer.writerow(headers)

        row = [dataset_name, algorithm_name, classifier_name, centroids, accuracy, precision, recall, f1] + list(best_params.values())
        writer.writerow(row)

    print(f"Najlepsze hiperparametry zapisano do pliku {filename} dla datasetu {dataset_name}, algorytmu {algorithm_name} i klasyfikatora {classifier_name}")


def load_best_params(filename):
    """
    Wczytuje dane z pliku CSV do listy słowników.
    """
    data = []
    try:
        with open(filename, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                row["Centroids"] = int(row["Centroids"])
                row["Accuracy"] = float(row["Accuracy"])
                row["Precision"] = float(row["Precision"])
                row["Recall"] = float(row["Recall"])
                row["F1"] = float(row["F1"])
                data.append(row)
    except FileNotFoundError:
        print(f"Plik {filename} nie istnieje.")
    return data


import statistics

from collections import defaultdict

def summarize_results(data):
    """
    Podsumowuje wyniki według algorytmu i klasyfikatora, obliczając średnie wartości metryk oraz centroidów.
    """
    metrics_by_classifier = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts = defaultdict(lambda: defaultdict(int))
    min_move_centroids = defaultdict(list)
    initial_centroids = defaultdict(list)

    for row in data:
        algorithm = row["Algorithm"]
        classifier = row["Classifier"]

        # Dodajemy wartości metryk do listy
        metrics_by_classifier[algorithm][classifier]["Accuracy"].append(row["Accuracy"])
        metrics_by_classifier[algorithm][classifier]["Precision"].append(row["Precision"])
        metrics_by_classifier[algorithm][classifier]["Recall"].append(row["Recall"])
        metrics_by_classifier[algorithm][classifier]["F1"].append(row["F1"])
        metrics_by_classifier[algorithm][classifier]["Centroids"].append(row["Centroids"])

        # Zbieramy centroidy dla "Min Move" klasyfikatora
        if classifier == "Min Move":
            min_move_centroids[algorithm].append(row["Centroids"])

        # Zbieramy początkowe centroidy (pierwszy parametr z best_params)
        if "initial_centroids" in row:
            initial_centroids[algorithm].append(int(row["initial_centroids"]))

        # Zwiększamy licznik dla danej pary algorytm-klasyfikator
        counts[algorithm][classifier] += 1

    # Obliczanie średnich wartości
    summary = defaultdict(lambda: defaultdict(dict))
    for algorithm, classifiers in metrics_by_classifier.items():
        for classifier, metrics in classifiers.items():
            for metric, values in metrics.items():
                summary[algorithm][classifier][metric + "_mean"] = sum(values) / len(values)
                summary[algorithm][classifier][metric + "_std"] = (sum((x - summary[algorithm][classifier][metric + "_mean"]) ** 2 for x in values) / len(values)) ** 0.5
                summary[algorithm][classifier][metric + "_max"] = max(values)

    # Obliczanie średnich centroidów dla Min Move
    min_move_avg = {algorithm: sum(centroids) / len(centroids) for algorithm, centroids in min_move_centroids.items()}
    
    # Obliczanie średnich początkowych centroidów
    initial_avg = {algorithm: sum(centroids) / len(centroids) for algorithm, centroids in initial_centroids.items()}

    return summary, min_move_avg, initial_avg


def print_summary(summary, min_move_avg, initial_avg):
    """
    Wyświetla podsumowanie wyników, w tym średnią, odchylenie standardowe i maksimum dla każdej metryki.
    """
    for algorithm, classifiers in summary.items():
        print(f"Algorytm: {algorithm}")
        for classifier, metrics in classifiers.items():
            print(f"  Klasyfikator: {classifier}")
            
            # Wypisanie wyników metryk z dodatkowymi statystykami
            for metric, value in metrics.items():
                if "_mean" in metric:
                    print(f"    Średnia {metric.split('_')[0]}: {value:.4f}")
                elif "_std" in metric:
                    print(f"    Odchylenie standardowe {metric.split('_')[0]}: {value:.4f}")
                elif "_max" in metric:
                    print(f"    Maksimum {metric.split('_')[0]}: {value:.4f}")
                else:
                    print(f"    {metric}: {value:.2f}")
            
            # Wypisanie średniej liczby centroidów dla klasyfikatora Min Move
            if classifier == "Min Move":
                avg_min_move_centroids = min_move_avg.get(algorithm, 0)
                print(f"    Średnia liczba centroidów dla klasyfikatora Min Move: {avg_min_move_centroids:.2f}")
            
            # Wypisanie średniej początkowej liczby centroidów
            avg_initial_centroids = initial_avg.get(algorithm, 0)
            print(f"    Średnia początkowa liczba centroidów: {avg_initial_centroids:.2f}")
            
            print()


# Przykładowe użycie
filename = "Heartbeat.csv"

# Wczytaj dane
data = load_best_params(filename)

# Podsumowanie wyników
summary, min_move_avg, initial_avg = summarize_results(data)

# Wyświetl podsumowanie
print_summary(summary, min_move_avg, initial_avg)
