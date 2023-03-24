import numpy as np
import matplotlib.pyplot as plt


def euclidian_distance(X1, X2):
    return np.sqrt(np.sum((X1-X2)**2))


class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # Lista de indices para cada cluster
        self.clusters = [[] for _ in range(self.K)]

        # los centroides
        self.centroides = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Inicializacion
        random_sample_indxs = np.random.choice(
            self.n_samples, self.K, replace=False)
        self.centroides = [self.X[idx] for idx in random_sample_indxs]

        for _ in range(self.max_iters):
            # asignar los datos al centroide mas cercano
            self.clusters = self._create_clusters(self.centroides)

            if self.plot_steps:
                self.plot()

            # calcualr los nuevos centroids
            centrois_old = self.centroides
            self.centroides = self._get_centroids(self.clusters)

            if self._is_converged(centrois_old, self.centroides):
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        return self._get_clusters_labels(self.clusters)

    def _get_clusters_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def _create_clusters(self, centroids):
        # Empezamos con una lista vacia para cada cluster
        clusters = [[] for _ in range(self.K)]
        for indice, muestra in enumerate(self.X):
            centroid_index = self._closest_centroid(muestra, centroids)
            clusters[centroid_index].append(indice)
        return clusters

    def _closest_centroid(self, muestra, centroids):
        # determinar la distancia de cada muestra a cada centroide
        # para despues obtener el centroide mas cercano
        distances = [euclidian_distance(muestra, punto) for punto in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # Asigna el valor medio de los clusters a los centroides
        centroids = np.zeros((self.K, self.n_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_index] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distancias entre viejos y nuevos centroides
        distances = [euclidian_distance(
            centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroides:
            ax.scatter(*point, marker='x', color='black', linewidth=2)

        plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()
