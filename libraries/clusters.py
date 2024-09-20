import numpy as np

# Zliczamy dla każdego clustra, liczbę wystąpień punktów należących do niego
def count_points_for_clusters(cluster_membership, n_clusters):
    # Zlicz występowanie każdej wartości (każdego klastra)
    counts = np.bincount(cluster_membership, minlength=n_clusters)

    return counts

def sum_probability_for_clusters(fuzzy_labels):
    summed_labels = np.sum(fuzzy_labels, axis=1)

    return summed_labels

def popularity_of_clusters(fuzzy_labels, n_clusters):

    cluster_membership = np.argmax(fuzzy_labels, axis=0)
    
    counts = count_points_for_clusters(cluster_membership, n_clusters)

    summed_labels = sum_probability_for_clusters(fuzzy_labels)
    
    # Iteracja przez każdy klaster
    for cluster in range(len(counts)):
        print(f"Cluster {cluster}: counts = {counts[cluster]}, summed_labels = {summed_labels[cluster]}, fcm per point {summed_labels[cluster]/counts[cluster]} ")