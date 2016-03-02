class DPMM(object):
	""" Class specifying the Dirichlet Process Mixture Model. 

        Initialise with DPMM(cluster_count, data_count, alpha, clusters, data, membership)

        Attributes: 

        cluster_count 	- integer detailing the number of clusters in the mixture model
		data_count 		- integer detailing the number of datapoints in the mixture model
		alpha		 	- concentration parameter, alpha, specifying probability of creating a new cluster 
		clusters		- list of each cluster, where each element is in the Distribution class
		data 			- numpy array where rows are data points
		membership 		- list detailing the membership of each data point in data. if membership[i] = j, then the ith row in data is a member of the jth distribution in clusters
		cluster_popn	- list specifying the number of datapoints in each cluster
    """
    def __init__(self, cluster_count, data_count, alpha, clusters, data, membership):
    	self.cluster_count = cluster_count
    	self.data_count = data_count
    	self.alpha = alpha
    	self.clusters = clusters
    	if data.shape[0] != data_count:
    		print "Provided data doesn't conform with claimed data count."
    		exit(1)
    	self.data = data
    	if len(membership) != data_count:
    		print "Membership array does not have the correct number of elements."
    		exit(1)
    	self.membership = np.asarray(membership)
    	self.cluster_popn = np.asarray([np.sum(np.equal(membership, cluster_idx)) for cluster_idx in range(cluster_count)])