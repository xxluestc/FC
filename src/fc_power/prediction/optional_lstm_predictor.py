"""Optional extension point; deep learning is not a dependency of the minimum route."""
def unavailable_reason():
    return 'Add GRU/LSTM only with identical time split and inputs; PyTorch is intentionally not a base dependency.'

