from distributions import *
from dpmm import *
from  collapsed_gibbs import *
import matplotlib.pyplot as plt
import cPickle

def plotClusters(model):
    plt.plot(model.data, 0*model.data, 'r+', markersize = 12)

    densityList = []
    for _ in range(100):
        (density, densityLocations) = getDensity(model)
        densityList.append(density)

    densityList = np.asarray(densityList)
    s = np.sort(densityList, 0)
    plt.plot(densityLocations, s[0], 'g', markersize = 12)
    plt.plot(densityLocations, np.mean(densityList, 0), 'r', markersize = 12)
    plt.plot(densityLocations, s[-1], 'b', markersize = 12)

    plt.xlabel('Input data')
    plt.ylabel('Probability')
    plt.title('DPMM')
    plt.grid(True)
    # plt.savefig("test.png")
    plt.show()  


def getDensity(model, densityLocations = np.arange(-10, 10, 0.1)):

    model.cluster_popn.append(model.alpha)
    mixingProps = scipy.stats.gamma.rvs(model.cluster_popn)
    mixingProps = mixingProps/sum(mixingProps)
    model.cluster_popn = model.cluster_popn[:-1]

    mu = []
    sigma = []
    density = np.zeros(densityLocations.shape)

    for cluster, idx in zip(model.clusters, range(len(model.clusters))):
        (mu_temp, sigma_temp) = cluster.draw_from_posterior()
        mu.append(mu_temp)
        sigma.append(sigma_temp)
        density += mixingProps[idx]*np.exp(-0.5*((densityLocations-mu[idx]) ** 2)/sigma[idx])/np.sqrt(2*np.pi*sigma[idx])
    
    return density, densityLocations


# Generate DPMM

cluster_count = 5
data_count = 200
alpha = 0.1
prior = {'d': 1., 'r': 1., 'v': 1., 'm': np.asarray(2.), 'S': np.asarray(1.)}
clusters = [Gaussian(np.asarray([]), prior) for _ in range(cluster_count)]
data = np.concatenate((np.asarray([[np.random.normal(4, 3)] for _ in range(data_count/4)]), np.asarray([[np.random.normal(-5, 1)] for _ in range(data_count/4)]), np.asarray([[np.random.normal(1, 2)] for _ in range(data_count/4)]), np.asarray([[np.random.normal(8, 0.5)] for _ in range(data_count/4)])))
membership = np.random.randint(0, cluster_count, data_count)

for data_point, clusterID in zip(data, membership):
    clusters[clusterID].add_data(data_point)

model = DPMM(cluster_count, data_count, alpha, clusters, data, membership, prior)

CollapsedGibbsSampling(model, 20)

plotClusters(model)

output = open('dpmmModel.pkl', 'wb')
cPickle.dump(model, output)

output.close()

