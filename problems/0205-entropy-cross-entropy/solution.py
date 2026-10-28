import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	
	P = np.asarray(P, dtype=float)
	Q = np.asarray(Q, dtype=float)

	P_hp = np.where(P > 0)

	hp = -1 * np.sum(P[P_hp] * np.log(P[P_hp]))

	P_hpq = np.where((P > 0) & (Q == 0))

	hpq = float('inf') if len(P_hpq[0]) != 0 else (-1 * np.sum(P * np.log(Q)))

	return round(hp, 6), round(hpq, 6)
