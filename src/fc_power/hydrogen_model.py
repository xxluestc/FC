"""Traceable theoretical hydrogen model."""
import numpy as np
FARADAY_C_PER_MOL=96485.33212; H2_G_PER_MOL=2.01588
def faraday_h2_g_s(current_a,n_cells=170.,faradaic_efficiency=1.):
    return n_cells*np.asarray(current_a)*H2_G_PER_MOL/(2*FARADAY_C_PER_MOL*faradaic_efficiency)

