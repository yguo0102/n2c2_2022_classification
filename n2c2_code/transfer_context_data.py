import pandas as pd
import sys
import os


dat_dir = sys.argv[1]

for data_split in ['train', 'dev', 'test']:
    dat = os.path.join(dat_dir, f'{data_split}.csv')
    if os.path.exists(dat):
        df = pd.read_csv(dat)

        print('ori:', len(df))
        df = df[df['label'].isin(['Disposition', 'Drug'])]
        print('new:', len(df))


        for att_typ in ['Action', 'Negation', 'Temporality', 'Certainty', 'Actor']:
            df.rename(columns={'label': 'prev_label', att_typ: 'label'}, inplace=True)
            head, tail = os.path.split(dat_dir.strip('/'))
            out_dir = f'./{head}/{tail}/{tail}_{att_typ}'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            df.to_csv(f'{out_dir}/{data_split}.csv', index=False)
            print(f'Done: {out_dir}/{data_split}.csv')
