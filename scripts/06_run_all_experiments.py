"""Run after local paths are configured; see README for commands."""
from pathlib import Path
def main():
    required=['data/processed/liu_vehicle_canonical_1s.csv','data/processed/power_demand_from_dynamics.csv','data/processed/prediction_results.csv','data/results/allocation/allocation_metrics.csv']
    missing=[x for x in required if not Path(x).exists()]
    if missing:raise SystemExit('Run numbered scripts in order; missing: '+', '.join(missing))
    print('All minimum-route artifacts exist. Run scripts/plot_results.py to regenerate figures.')
if __name__=='__main__':main()

