import pandas as pd
import os, sys
import re
import glob

targets = ['Action', 'Actor', 'Certainty', 'Negation', 'Temporality']

def read_data(pred_data):
    pred_df = pd.read_csv(pred_data, sep='\t')
    ret_map = {}
    for file_id, pred in zip(pred_df['file'], pred_df['prediction']):
        file_id.strip('/')
        assert file_id not in ret_map
        ret_map[file_id] = pred
    return ret_map


if __name__ == '__main__':
    data_dir = sys.argv[1]
    test_data = sys.argv[2]
    pred_data_dir = sys.argv[3]
    out_dir = sys.argv[4]

    test_df = pd.read_csv(test_data)

    pred_data = f'{pred_data_dir}/Event.txt'
    pred_df = pd.read_csv(pred_data, sep='\t')
    assert len(test_df) == len(pred_df)

    file_event_map = {}
    for fe_id, pred in zip(test_df['file'], pred_df['prediction']):
        file_id, TID = fe_id.split('_')
        file_id.strip('/')
        if file_id not in file_event_map:
            file_event_map[file_id] = {}
        assert TID not in file_event_map
        file_event_map[file_id][TID] = pred

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_other_map = {}
    for other in targets:
        file_other_map[other] = read_data(f'{pred_data_dir}/{other}_fileid.txt')

    for file_id, event_map in file_event_map.items():
        e_count = 0
        a_count = 0
        with open(f'{out_dir}/{file_id}.ann', 'w') as fw:
            with open(f'{data_dir}/{file_id}.ann', 'r') as fr:
                for line in fr:
                    if re.search('^T', line) != None:
                        fw.write(f'{line}')

            for TID, pred in event_map.items():
                assert 'E' not in TID
                e_count += 1
                EID = f'E{e_count}'
                fw.write(f'{EID}\t{pred}:{TID}\n')

                if pred == 'Disposition':
                    full_id = f'{file_id}_{TID}'
                    for other in targets:
                        if full_id in file_other_map[other]:
                            a_count += 1
                            AID = f'A{a_count}'
                            #A4^ITemporality E7 Past
                            fw.write(f'{AID}\t{other} {EID} {file_other_map[other][full_id]}\n')
    print('Done! output is {}'.format(out_dir))

