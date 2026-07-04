"""Longitudinal bus dynamics with explicit force components."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class VehicleParams:
    mass_kg:float=16400.; drag_coefficient:float=.62; frontal_area_m2:float=7.6
    rolling_resistance:float=.015; rotational_mass_factor:float=1.08
    air_density_kg_m3:float=1.225; gravity_mps2:float=9.80665; road_grade_rad:float=0.

def force_power(speed_mps,acceleration_mps2,p:VehicleParams=VehicleParams()):
    v=np.asarray(speed_mps,float); a=np.asarray(acceleration_mps2,float); alpha=p.road_grade_rad
    f_air=.5*p.air_density_kg_m3*p.drag_coefficient*p.frontal_area_m2*v**2
    f_roll=p.mass_kg*p.gravity_mps2*p.rolling_resistance*np.cos(alpha)*(v>0)
    f_grade=p.mass_kg*p.gravity_mps2*np.sin(alpha)*np.ones_like(v)
    f_acc=p.rotational_mass_factor*p.mass_kg*a
    f_trac=f_air+f_roll+f_grade+f_acc; p_wheel=f_trac*v/1000
    return {'f_air_n':f_air,'f_roll_n':f_roll,'f_grade_n':f_grade,'f_acc_n':f_acc,'f_trac_n':f_trac,'p_wheel_kw':p_wheel}

