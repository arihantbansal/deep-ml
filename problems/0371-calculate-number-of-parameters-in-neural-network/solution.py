def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""

	params = 0

	for layer in layers:
		if layer["type"] == "dense":
			i = layer["input_size"]
			o = layer["output_size"]
			b = layer.get("bias", True)
			
			params += i * o
			params += o if b else 0
		else:
			i = layer["in_channels"]
			o = layer["out_channels"]
			b = layer.get("bias", True)
			k = layer["kernel_size"]

			params += i * o * k * k 
			params += o if b else 0
		
	return params