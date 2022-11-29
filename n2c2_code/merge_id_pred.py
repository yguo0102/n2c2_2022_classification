import pandas as pd
import sys

test_data = sys.argv[1]
pred_data = sys.argv[2]
out_file = sys.argv[3]

test_df = pd.read_csv(test_data)
pred_df = pd.read_csv(pred_data, usecols=['prediction'], dtype=str, sep='\t')
assert len(test_df) == len(pred_df), f'Error: {len(test_df)} != {len(pred_df)}'

out_df = {'file':test_df['file'].values.tolist(), 'prediction':pred_df['prediction'].values.tolist()}
out_df = pd.DataFrame(out_df)
out_df.to_csv(out_file, index=False, sep='\t')
