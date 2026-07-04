"""Canonical preprocessing for Liu's 21UBE0022 vehicle CSV files."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

COLUMN_MAP={
    '上报时间':'timestamp','车速':'speed_kmh','电池SOC':'soc_pct','目标功率':'target_power_kw',
    'DCDC输入电压':'fc_voltage_v','DCDC输入电流':'fc_current_a',
    'DCDC输出电压':'dcdc_output_voltage_v','DCDC输出电流':'dcdc_output_current_a',
    '电池电压':'battery_voltage_v','电池电流':'battery_current_a',
    '电机电压':'motor_voltage_v','电机电流':'motor_current_a',
    '单次耗氢量(Kg)':'single_h2_kg','累计耗氢量(Kg)':'cumulative_h2_kg',
    '可加载功率':'loadable_power_kw','整车累计行驶里程':'odometer_km',
    '节电压平均值':'mean_cell_voltage_v','最低节电压':'min_cell_voltage_v','最高节电压':'max_cell_voltage_v',
}

def load_liu_csv(path:Path)->pd.DataFrame:
    df=pd.read_csv(path,encoding='utf-8-sig',low_memory=False)
    available={k:v for k,v in COLUMN_MAP.items() if k in df.columns}
    out=df[list(available)].rename(columns=available).copy(); out['source_file']=path.name
    out['timestamp']=pd.to_datetime(out['timestamp'],errors='coerce')
    for c in out.columns.difference(['timestamp','source_file']): out[c]=pd.to_numeric(out[c],errors='coerce')
    return out.dropna(subset=['timestamp'])

def canonicalize(files:list[Path],target_dt_s:float=1.,gap_s:float=10.)->tuple[pd.DataFrame,dict]:
    raw=pd.concat([load_liu_csv(f) for f in files],ignore_index=True).sort_values('timestamp')
    duplicates=int(raw.timestamp.duplicated().sum()); numeric=raw.select_dtypes(include='number').columns.tolist()
    # Duplicate packets at the same timestamp are collapsed by numeric mean.
    raw=raw.groupby('timestamp',as_index=False)[numeric].mean().sort_values('timestamp')
    dt=raw.timestamp.diff().dt.total_seconds(); raw['segment_id']=(dt.isna()|(dt>gap_s)).cumsum()-1
    pieces=[]
    for sid,g in raw.groupby('segment_id'):
        x=g.set_index('timestamp').drop(columns='segment_id').resample(f'{target_dt_s}s').mean()
        # Short within-run gaps only. No interpolation across segment boundaries.
        x=x.interpolate('time',limit=max(1,int(gap_s/target_dt_s)-1, ),limit_area='inside'); x['segment_id']=sid; pieces.append(x.reset_index())
    out=pd.concat(pieces,ignore_index=True).sort_values('timestamp')
    out['speed_mps']=out.speed_kmh/3.6
    out['acceleration_mps2']=out.groupby('segment_id').speed_mps.diff()/target_dt_s
    out['fc_input_power_kw']=out.fc_voltage_v*out.fc_current_a/1000
    out['dcdc_output_power_kw']=out.dcdc_output_voltage_v*out.dcdc_output_current_a/1000
    out['battery_power_kw_raw_sign']=out.battery_voltage_v*out.battery_current_a/1000
    out['motor_power_kw_raw_sign']=out.motor_voltage_v*out.motor_current_a/1000
    info={'input_files':len(files),'raw_rows':len(pd.concat([load_liu_csv(f) for f in files],ignore_index=True)),
          'duplicate_timestamps':duplicates,'unique_rows':len(raw),'processed_rows':len(out),'segments':int(out.segment_id.nunique()),
          'target_dt_s':target_dt_s,'gap_threshold_s':gap_s,'start':str(out.timestamp.min()),'end':str(out.timestamp.max()),
          'speed_nonmissing':int(out.speed_kmh.notna().sum()),'speed_nonzero':int((out.speed_kmh>0).sum())}
    return out,info

