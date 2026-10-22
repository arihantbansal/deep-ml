import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	
	def sigmoid(x):
		return 1 / (1 + math.exp(-x))

	loss = 0
	probabilities = []
	for i, feature in enumerate(features):
		z = bias + sum(a * b for a, b in zip(feature, weights))
		p = sigmoid(z)
		probabilities.append(p)

		loss += ((p - labels[i]) ** 2)
	
	mse = loss / len(features)

	return probabilities, mse