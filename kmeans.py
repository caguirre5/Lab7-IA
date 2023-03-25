import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, K=7, max_iters=10, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # Lista de indices para cada cluster
        self.clusters = [[] for _ in range(self.K)]
        self.centroides = None

    
    def train(self, X):
        # inicializamos los centroides
        self.centroides = np.random.rand(self.K,X.shape[1])
        centroids_old = self.centroides.copy()

        for _ in range(self.max_iters):
            distance=None
            for sample in range(1,self.K):
                distance = self.euclidian_distance
            samples = np.argmin(distance,axis=1) 
            # asignar los datos al centroide mas cercano
            for sample in set(samples):
                self.centroides[sample,:] = self._get_clusters_labels(self, X, samples, sample)
            #Si los centroides convergen, 
            if self._is_converged(centroids_old, self.centroides):
                break
        return None


    def predict(self, X):
        distance = None
        for sample in range(1,self.k):
            distance = self.euclidian_distance
        return self._get_centroids(distance)
        

    def _get_clusters_labels(self, X,samples, sample):
        clusters=7
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return np.mean(X[samples == sample,:],axis=0)

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
        distances = [self.euclidian_distance(muestra, punto) for punto in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, distance):
        # Asigna el valor medio de los clusters a los centroides
        centroids = np.zeros((self.K, self.n_features))
        
        return np.argmin(distance,axis=1)

    def _is_converged(self, centroids_old, centroids):
        # distancias entre viejos y nuevos centroides
        result = np.linalg.norm(centroids - centroids_old) < 1e-3
        return result
    
    def euclidian_distance(self, X):
        return np.linalg.norm(X - self.centroides[0,:],axis=1).reshape(-1,1)

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroides:
            ax.scatter(*point, marker='x', color='black', linewidth=2)

        plt.show()