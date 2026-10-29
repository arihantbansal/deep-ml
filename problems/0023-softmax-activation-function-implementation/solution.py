import math

def softmax(scores: list[float]) -> list[float]:

    c = max(scores)
    exps = [math.exp(score - c) for score in scores]
    s = sum(exps)
    return [exp / s for exp in exps]