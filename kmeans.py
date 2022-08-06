import random
class KMeansClusterClassifier():
    def __init__(self,n_cluster = 3):
        '''Initializes the object'''
        self.n_cluster = n_cluster
        self.n_iter = 500
        self.inertia_ = 0
    def update_labels(self,X):
        '''Updates predictions based on new centroid values'''
        self.inertia_ = 0
        num_features = len(X[0])
        features = []
        for i in range(num_features):
            features.append(list(zip(*X))[i]) # Column transform
        labels   = []
        for i in range(len(X)):
            min_distance = 1e9
            label = -1
            for k in range(self.n_cluster):
                cluster_distance = 0
                for j in range(num_features):
                    cluster_distance += pow(features[j][i] - self.centroids[k][j],2)
                    # (feature j of element i)  -
                cluster_distance = pow(cluster_distance,0.5)
                if cluster_distance < min_distance: # Decides which centroid to predict
                    min_distance = cluster_distance
                    label = k
            self.inertia_+=min_distance
            labels.append(label)
        return labels
    def create_centroids(self,X): 
        '''Creates random values for centroids at the beginning of training process'''
        num_feature = len(X[0])
        weights = [[0]*2]*num_feature # [[min,max]*num_features]
        for i in range(num_feature): # for every feature
            weights[i] = [min(list(zip(*X))[i]),max(list(zip(*X))[i])] 
            # weights[i][0] is equal to min of feature i
        centroids = []
        for i in range(self.n_cluster): 
            centroid = []
            random.seed(i)
            for k in range(num_feature):
                centroid.append(weights[k][0]  + random.random()*(weights[k][1]-weights[k][0])) 
            centroids.append(centroid) # random values  = min + U(0,1)(max-min)
        return centroids
    def update_centroids(self,X,labels) :
        '''Updates centroid values at every iteration based on new label values'''
        num_features = len(X[0])
        centroid_k = []
        for _ in range(self.n_cluster):
            centroid_k.append([0]*num_features)
        count = [0]*self.n_cluster
        for idx,element in enumerate(X):
            for k in range(self.n_cluster):
                if labels[idx] == k:
                    for j in range(num_features):
                        centroid_k[k][j] += element[j]
                    count[k]+=1
        for i in range(self.n_cluster):
            if count[i] != 0:
                centroid_k[i] = [x / count[i] for x in centroid_k[i]]
        return centroid_k
        
    def fit(self,X):
        '''Training function.'''
        '''Creates centroid values and labels. After that for n_iter times, trains the model'''
        self.centroids = self.create_centroids(X)
        labels = self.update_labels(X)
        for _ in range(self.n_iter):
            self.centroids = self.update_centroids(X,labels)
            labels = self.update_labels(X)
        return labels
    def predict(self,X):
        '''Uses update_labels method since centroid values are set and new predictions
        will be made based on new X values'''
        return self.update_labels(X)
