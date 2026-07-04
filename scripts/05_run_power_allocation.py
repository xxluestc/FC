from pathlib import Path
import argparse,sys,time
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fc_power.power_allocation.mpc_allocator import choose
from fc_power.battery_model import next_soc,throughput_cost

def run(name,demand,source_idx,predmap,actions,h2,deg,H):
 soc=.70; prev=0; dwell=15; rows=[]; tic=time.perf_counter()
 for q,(k,p) in enumerate(zip(source_idx,demand)):
  hh=min(H,len(demand)-q)
  if name=='instant':preview=np.array([p])
  elif name=='constant':preview=np.repeat(p,hh)
  elif name=='perfect':preview=demand[q:q+hh]
  else:preview=np.r_[p,[predmap.get((k,h),p) for h in range(1,hh)]]
  tier=choose(preview,soc,prev,dwell,actions,h2,deg); pfc=actions[tier]; pbat=p-pfc; sn=float(next_soc(soc,pbat))
  rows.append({'step':q,'source_index':k,'strategy':name,'demand_kw':p,'p_fc_kw':pfc,'p_bat_kw':pbat,'soc':sn,'tier':tier,'h2_g':h2[tier],'deg_cost':deg[tier]})
  dwell=min(15,dwell+1) if tier==prev else 1; prev=tier; soc=sn
 tr=pd.DataFrame(rows); sw=int(tr.tier.diff().fillna(0).ne(0).sum()); m={'strategy':name,'n':len(tr),'h2_kg':tr.h2_g.sum()/1000,'deg_cost_sum':tr.deg_cost.sum(),'soc_final':tr.soc.iloc[-1],'soc_error':tr.soc.iloc[-1]-.70,'battery_throughput_kwh':tr.p_bat_kw.abs().sum()/3600,'switch_count':sw,'fc_total_variation_kw':tr.p_fc_kw.diff().fillna(0).abs().sum(),'runtime_s':time.perf_counter()-tic}; return tr,m
def main():
 p=argparse.ArgumentParser(); p.add_argument('--vehicle',type=Path,required=True); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--stack-map',type=Path,required=True); p.add_argument('--out-dir',type=Path,required=True); a=p.parse_args(); v=pd.read_csv(a.vehicle); pr=pd.read_csv(a.predictions); sm=pd.read_csv(a.stack_map)
 cp=pr[pr.method.eq('condition_rf_corrected')]; origins=np.sort(cp.origin_index.unique()); origin_set=set(origins); runs=[]; cur=[]
 for x in origins:
  if cur and x!=cur[-1]+1:runs.append(cur); cur=[]
  cur.append(int(x))
 if cur:runs.append(cur)
 seq=max(runs,key=len)[:3600]; raw=v.loc[seq,'p_dem_measured_kw'].to_numpy(); lo=-75.; hi=120+sm.stack_power_kw.max(); demand=np.clip(raw,lo,hi)
 predmap={(int(x.origin_index),int(x.horizon_s)):float(np.clip(x.power_pred_kw,lo,hi)) for x in cp.itertuples()}; actions=sm.stack_power_kw.to_numpy(); h2=sm.faraday_h2_g_s.to_numpy(); deg=sm.performance_loss_cost_normalized.to_numpy()
 a.out_dir.mkdir(parents=True,exist_ok=True); mets=[]; trs=[]
 for name,H in [('instant',1),('constant',10),('perfect',10),('predicted',10)]:tr,m=run(name,demand,seq,predmap,actions,h2,deg,H); trs.append(tr); mets.append(m); print(m)
 pd.concat(trs).to_csv(a.out_dir/'allocation_trajectory.csv',index=False); pd.DataFrame(mets).to_csv(a.out_dir/'allocation_metrics.csv',index=False); pd.DataFrame({'source_index':seq,'raw_demand_kw':raw,'feasible_demand_kw':demand}).to_csv(a.out_dir/'test_demand.csv',index=False)
if __name__=='__main__':main()
