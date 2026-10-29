import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""

	def value(func: str, a: float) -> float:
		match func:
			case 'square':
				return a ** 2
			case 'sin':
				return np.sin(a).item()
			case 'exp':
				return np.exp(a).item()
			case 'log':
				return np.log(a).item()
			case _:
				return 0
	
	def deriv_value(func: str, b: float) -> float:
		match func:
			case 'square':
				return 2 * b
			case 'sin':
				return np.cos(b).item()
			case 'exp':
				return np.exp(b).item()
			case 'log':
				return 1 / b
			case _:
				return 0
	
	values = [x, value(functions[-1], x)]

	for f in functions[:-1][::-1]:
		values.append(value(f, values[-1]))

	out = 1
	l = len(values)

	for i, f in enumerate(functions):
		out = out * deriv_value(f, values[l - 2 - i])

	return out
		