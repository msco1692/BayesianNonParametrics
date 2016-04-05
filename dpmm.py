import numpy as np

class DPMM(object):
    """ Class specifying the Dirichlet Process Mixture Model. 

        Initialise with DPMM(cluster_count, data_count, alpha, clusters, data, membership)

        Attributes: 

        cluster_count     - integer detailing the number of clusters in the mixture model
        data_count         - integer detailing the number of datapoints in the mixture model
        alpha             - concentration parameter, alpha, specifying probability of creating a new cluster 
        clusters        - list of each cluster, where each element is in the Distribution class
        data             - numpy array where rows are data points
        membership         - list detailing the membership of each data point in data. if membership[i] = j, then the ith row in data is a member of the jth distribution in clusters
        cluster_popn    - list specifying the number of datapoints in each cluster

        Methods:

        conditional_prob(self, data_point)    - outputs a numpy array of the relative probabilities that a given data point belongs with the others in each cluster
    """
    def __init__(self, cluster_count, data_count, alpha, clusters, data, membership, prior):
        self.data_count = data_count
        self.alpha = alpha
        # Clusters should include an empty cluster at the end of the list which is ignored in cluster_count
        self.clusters = clusters

        self.cluster_count = cluster_count# if data.shape[0] != data_count:
     #        print "Provided data doesn't conform with claimed data count."
     #        exit(1)
        self.data = data
        self.prior = prior

        if len(membership) != data_count:
            print "Membership array does not have the correct number of elements."
            exit(1)
        self.membership = np.asarray(membership)
        self.cluster_popn = [np.sum(np.equal(membership, cluster_idx)) for cluster_idx in range(cluster_count)]

    def conditional_prob(self, data_point):
        "Outputs a numpy array of the relative probabilities that a given data point belongs with the others in each cluster"

        p = np.log(self.cluster_popn)
        p = np.append(p, np.log(self.alpha))
        for cluster, cluster_id in zip(self.clusters, range(len(self.clusters))):
           p[cluster_id] += cluster.log_pred(data_point)
        p = np.exp(p)
        return p/np.sum(p)