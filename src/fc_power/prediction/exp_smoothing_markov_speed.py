"""Damped local trend with operating-state residual correction."""
import numpy as np

class DampedTrendMarkov:
    def __init__(self,damping=.82):self.damping=damping; self.table={}; self.default=0.
    @staticmethod
    def state(v,a):return (int(np.clip(v//5,0,20)),int(np.digitize(a,[-.5,-.1,.1,.5])))
    def fit(self,v,a,one_step_residual):
        buckets={}
        for vi,ai,ri in zip(v,a,one_step_residual):buckets.setdefault(self.state(vi,ai),[]).append(ri)
        self.table={k:float(np.mean(x)) for k,x in buckets.items()}; self.default=float(np.mean(one_step_residual)); return self
    def predict_one(self,v_history,horizon):
        v=float(v_history[-1]); trend=float(np.mean(np.diff(v_history[-10:]))) if len(v_history)>1 else 0.; out=[]
        for h in range(1,horizon+1):
            corr=self.table.get(self.state(v,trend),self.default); step=(self.damping**(h-1))*trend+corr; v=max(0.,v+step); out.append(v)
        return np.asarray(out)

