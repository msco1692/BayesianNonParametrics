from distributions import *
from dpmm import *
from  collapsed_gibbs import *

# Generate DPMM

cluster_count = 5
data_count = 20
alpha = 0.1
prior = {'d': 1., 'r': 1., 'v': 1., 'm': np.asarray(2.), 'S': np.asarray(1.)}
clusters = [Gaussian(np.asarray([]), prior) for _ in range(cluster_count)]
data = np.concatenate((np.asarray([[np.random.normal(-5, 1)] for _ in range(data_count/2)]), np.asarray([[np.random.normal(2, 2)] for _ in range(data_count/2)])))
membership = np.random.randint(0, cluster_count, data_count)

for data_point, clusterID in zip(data, membership):
    clusters[clusterID].add_data(data_point)

model = DPMM(cluster_count, data_count, alpha, clusters, data, membership, prior)

CollapsedGibbsSampling(model, 10)