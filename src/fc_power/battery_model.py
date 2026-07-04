"""Simple Rint battery model traced to Han's energy-management equations."""
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class BatteryParams:
    energy_kwh:float=37.; nominal_voltage_v:float=650.; resistance_ohm:float=.3
    charge_efficiency:float=.95; discharge_efficiency:float=.95
    soc_min:float=.30; soc_max:float=.90; charge_power_limit_kw:float=-75.; discharge_power_limit_kw:float=120.
    @property
    def capacity_ah(self):return self.energy_kwh*1000/self.nominal_voltage_v
def current_a(power_kw,p:BatteryParams=BatteryParams()):
    disc=np.maximum(p.nominal_voltage_v**2-4*p.resistance_ohm*np.asarray(power_kw)*1000,0)
    return (p.nominal_voltage_v-np.sqrt(disc))/(2*p.resistance_ohm)
def next_soc(soc,power_kw,dt_s=1.,p:BatteryParams=BatteryParams()):
    pw=np.asarray(power_kw); effective=np.where(pw>=0,pw/p.discharge_efficiency,pw*p.charge_efficiency)
    return np.asarray(soc)-effective*dt_s/(p.energy_kwh*3600)
def throughput_cost(power_kw,dt_s=1.):return np.abs(np.asarray(power_kw))*dt_s/3600

