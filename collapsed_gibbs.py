import numpy as np

def CollapsedGibbsSampling(model, num_iterations):
	"""Performs a specified number of iterations of the Collapsed Gibbs Sampling algorithm using the model specified."""

	# For each iteration
	for iter_id in range(num_iterations):

		# For each data point in the model
		for data_id in range(len(model.data)):

			# Remove data point from its current cluster
			data_point = model.data[data_id]
			cluster_id = model.membership[data_id]
			model.cluster_popn[cluster_id] -= 1
			model.clusters[cluster_id].rem_data(data_point)

			# If the cluster is now empty delete it
			if model.cluster_popn[cluster_id] == 0:
				model.cluster_count -= 1
				del model.clusters[cluster_id]
				del model.cluster_popn[cluster_id]
				tmp_idx = np.where(model.membership > cluster_id)
				model.membership[tmp_idx] -= 1

			# Generate numpy array of relative probabilities based on prior for membership and current datapoints in each cluster
			p = np.asarray(model.conditional_prob(data_point))

			# Check that p is normalised
			p = p/np.sum(p)

			# Generate cumulative probability mass function
			p = np.cumsum(p)
			rand_val = np.random.random()

			# Select new cluster based on probabilities
			new_cluster_id = np.sum(np.greater(rand_val, p))

			# If a new cluster was chosen then create it. For convenience an empty cluster will always be at the end of the list. Thus we only have to create a copy of this cluster for future iterations, without storing the prior.
			if new_cluster_id == model.cluster_count:
				model.cluster_count += 1
				model.clusters.append(model.cluster[-1].copy())
				np.append(model.cluster_popn, 0)

			# Add data point to its new cluster
			model.membership[data_id] = new_cluster_id
			model.cluster_popn[new_cluster_id] += 1
			model.clusters[new_cluster_id].add_data(data_point)