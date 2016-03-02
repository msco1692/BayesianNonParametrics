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
				del model.cluster[cluster_id]
				del model.cluster_popn[cluster_id]
				tmp_idx = np.where(model.membership > cluster_id)
				model.membership[tmp_idx] -= 1

			# Generate list of conditional probabilities based on prior for membership
			p = model.conditional_prob(data_point)

			