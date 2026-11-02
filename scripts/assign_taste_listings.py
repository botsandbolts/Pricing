"""Assign new daily listings with the frozen offline model, no fit calls."""
import argparse,json
from pathlib import Path
import joblib,pandas as pd
from pricing.taste_ranking import predict_new_profiles

def main():
    p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('listings');p.add_argument('--output',required=True);a=p.parse_args()
    bundle=joblib.load(a.bundle) # Only load the trusted build artifact, never a user upload.
    with Path(a.output).open('w') as out:
        for frame in pd.read_json(a.listings,lines=True,chunksize=1024):
            m,d=predict_new_profiles(frame,bundle)
            for idx,(weights,distances) in enumerate(zip(m,d)):
                out.write(json.dumps({'id':str(frame.iloc[idx].get('row_id',frame.index[idx])),'memberships':weights.tolist(),'unassigned':max(0.,1-float(weights.sum())),'medoid_l1':distances.tolist(),'nearest_medoid_profiles':distances.argsort().tolist(),'distance_units':'L1 sum of absolute differences in frozen scaled features; not dollars'})+'\n')
if __name__=='__main__':main()
