import numpy as np
from fc_power.battery_model import next_soc,throughput_cost

def choose(preview,soc,prev,dwell,actions,h2,deg,beam_width=8,min_dwell=15,soc_ref=.70):
    beam=[(0.,soc,prev,dwell,prev)]
    for j,pdem in enumerate(preview):
        cand=[]
        for cost,s,t,dw,first in beam:
            allowed=[t] if dw<min_dwell else range(max(0,t-1),min(len(actions),t+2))
            for nt in allowed:
                pfc=actions[nt]; pbat=pdem-pfc; sn=float(next_soc(s,pbat))
                if not(-75<=pbat<=120 and .30<=sn<=.90):continue
                # Charge-sustaining ECMS-like feedback: a SOC deficit raises
                # the desired FC contribution before the short horizon can see
                # the end of the trip.
                pref=np.clip(max(pdem,0)+1200*(soc_ref-s),actions.min(),actions.max())
                c=.45*h2[nt]/max(h2.max(),1e-9)+deg[nt]+1.5*abs(pbat)/120+3*abs(pfc-pref)/max(actions.max(),1)+.5*abs(sn-soc_ref)/.1+.08*(nt!=t)+.005*abs(pfc-actions[t])/max(np.diff(actions).max(),1)
                nd=min_dwell if nt==t and dw>=min_dwell else dw+1 if nt==t else 1
                cand.append((cost+c,sn,nt,nd,nt if j==0 else first))
        if not cand:return prev
        beam=sorted(cand,key=lambda z:z[0]+50*max(0,abs(z[1]-soc_ref)-.02)**2)[:beam_width]
    return min(beam,key=lambda z:z[0]+50*max(0,abs(z[1]-soc_ref)-.02)**2)[4]
