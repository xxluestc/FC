"""Audit source directories without copying private raw data."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fc_power.data.audit_data import inventory,csv_header,match_signals

def main():
    p=argparse.ArgumentParser(); p.add_argument('--han',type=Path,required=True); p.add_argument('--liu',type=Path,required=True); p.add_argument('--li',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    result={}
    for name,root in [('Han',a.han),('Liu',a.liu),('Li',a.li)]:
        csvs=list(root.rglob('*.csv')); samples=[]
        for f in csvs[:20]:
            try:
                cols=csv_header(f); samples.append({'path':str(f.relative_to(root)),'n_columns':len(cols),'signals':match_signals(cols)})
            except Exception as e:samples.append({'path':str(f.relative_to(root)),'error':str(e)})
        result[name]={'root':str(root),'inventory':inventory(root),'csv_samples':samples}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()

