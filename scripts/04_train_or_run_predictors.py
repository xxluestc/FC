from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fc_power.prediction.condition_aware_predictor import build_model
from fc_power.prediction.exp_smoothing_markov_speed import DampedTrendMarkov
from fc_power.vehicle_dynamics import VehicleParams,force_power

H=15; N=30
def feature(d,i):
    lags=[0,1,2,3,5,10,20,30]; v=d.speed_smooth_mps.to_numpy(); a=d.acceleration_smooth_mps2.to_numpy(); p=d.p_dem_measured_kw.to_numpy()
    vw=v[i-9:i+1]; aw=a[i-9:i+1]; pw=p[i-9:i+1]
    mode=[v[i]<.3,a[i]>.3,a[i]<-.3,abs(a[i])<=.1,v[i]>13.9]
    return np.r_[v[i-np.array(lags)],a[i-np.array(lags)],p[i],np.mean(pw),np.std(pw),np.mean(vw),np.std(vw),np.mean(aw),np.std(aw),mode]
def build(d):
    seg=d.segment_id.to_numpy(); idx=[]
    for i in range(N,len(d)-H):
        if seg[i-N]==seg[i+H]:idx.append(i)
    idx=np.asarray(idx); X=np.vstack([feature(d,i) for i in idx]); v=d.speed_smooth_mps.to_numpy(); p=d.p_dem_measured_kw.to_numpy()
    Y=np.vstack([v[i+1:i+H+1] for i in idx]); YP=np.vstack([p[i+1:i+H+1] for i in idx]); return idx,X,Y,YP
def speed_to_power(v0,vpred,residual0,traction_slope,traction_intercept,brake_slope,brake_intercept):
    seq=np.r_[v0,vpred]; acc=np.diff(seq); f=force_power(vpred,acc,VehicleParams())['p_wheel_kw']; dyn=np.where(f>=0,traction_intercept+traction_slope*f,brake_intercept+brake_slope*f)
    return dyn+residual0*np.exp(-np.arange(1,len(vpred)+1)/5.)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--dynamics-metrics',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--metrics',type=Path,required=True); a=ap.parse_args(); d=pd.read_csv(a.input,parse_dates=['timestamp']); idx,X,Y,YP=build(d)
 n=len(d); train=idx<int(.7*n); val=(idx>=int(.7*n))&(idx<int(.85*n)); test=idx>=int(.85*n)
 # cap training size deterministically without disturbing temporal test.
 tr=np.flatnonzero(train); tr=tr[np.linspace(0,len(tr)-1,min(100000,len(tr))).astype(int)]
 rf=build_model(); rf.fit(X[tr],np.c_[Y[tr],YP[tr]])
 v=d.speed_smooth_mps.to_numpy(); acc=d.acceleration_smooth_mps2.to_numpy(); one=[]
 for i in idx[train]:
  trend=np.mean(np.diff(v[i-9:i+1])); base=max(0,v[i]+trend); one.append(v[i+1]-base)
 mk=DampedTrendMarkov().fit(v[idx[train]],acc[idx[train]],np.asarray(one))
 dm=json.loads(a.dynamics_metrics.read_text()); tm=dm['models']['traction']; bm=dm['models']['braking']; residual=d.dynamics_residual_kw.to_numpy(); actual_p=d.p_dem_measured_kw.to_numpy()
 rows=[]; testpos=np.flatnonzero(test); rf_out=rf.predict(X[testpos]); rf_pred=rf_out[:,:H]; rf_power=rf_out[:,H:]
 for q,pos in enumerate(testpos):
  i=idx[pos]; preds={'persistence':np.repeat(v[i],H),'damped_markov':mk.predict_one(v[i-N:i+1],H),'condition_rf':np.maximum(0,rf_pred[q])}
  for name,vp in preds.items():
   pp=speed_to_power(v[i],vp,residual[i],tm['slope'],tm['intercept_kw'],bm['slope'],bm['intercept_kw'])
   for h in range(H):rows.append({'origin_index':i,'target_index':i+h+1,'horizon_s':h+1,'method':name,'speed_pred_mps':vp[h],'speed_actual_mps':Y[pos,h],'power_pred_kw':pp[h],'power_actual_kw':actual_p[i+h+1]})
  # Same state-aware model, but a direct power head corrects the known flat-road
  # dynamics residual (grade/accessories/driver torque). Speed predictions remain
  # explicit and are still available for physical consistency checks.
  for h in range(H):rows.append({'origin_index':i,'target_index':i+h+1,'horizon_s':h+1,'method':'condition_rf_corrected','speed_pred_mps':rf_pred[q,h],'speed_actual_mps':Y[pos,h],'power_pred_kw':rf_power[q,h],'power_actual_kw':actual_p[i+h+1]})
 out=pd.DataFrame(rows); a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
 metrics=[]
 for name,g in out.groupby('method'):
  for hh in [1,5,10,15]:
   z=g[g.horizon_s<=hh]; pa=z.pivot(index='origin_index',columns='horizon_s',values='power_actual_kw').mean(1); pp=z.pivot(index='origin_index',columns='horizon_s',values='power_pred_kw').mean(1)
   va=z[z.horizon_s==hh].speed_actual_mps; vp=z[z.horizon_s==hh].speed_pred_mps
   metrics.append({'method':name,'horizon_s':hh,'speed_mae_mps':mean_absolute_error(va,vp),'speed_rmse_mps':mean_squared_error(va,vp)**.5,'mean_power_mae_kw':mean_absolute_error(pa,pp),'mean_power_rmse_kw':mean_squared_error(pa,pp)**.5,'window_energy_mae_kwh':mean_absolute_error(pa*hh/3600,pp*hh/3600)})
 md=pd.DataFrame(metrics); a.metrics.parent.mkdir(parents=True,exist_ok=True); md.to_csv(a.metrics,index=False); print(md.to_string(index=False))
if __name__=='__main__':main()
