import numpy as np
def predict(current_value,horizon:int): return np.repeat(np.asarray(current_value)[...,None],horizon,axis=-1)

