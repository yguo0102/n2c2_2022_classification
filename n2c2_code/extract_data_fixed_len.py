from brat_parser import get_entities_relations_attributes_groups
import math
import pandas as pd
import glob
import string
import os
import sys
import re

MAX_LEN = 50

def find_attr(attributes, attr_type, ent_id):
    event_id = ent_id.replace('T', 'E')

    for act_id, attrs in attributes.items():
        if attrs.target == event_id and attrs.type == attr_type:
            return attrs.values[0]
    return 'NA'

def adjust_prev_window(text, index, n_words):
    if index == 0:
        return 0

    while n_words > 0 and index > 0:
        index -= 1
        if text[index] is ' ':
            n_words -= 1

    return index


def adjust_next_window(text, index, n_words):
    if index >= len(text)-1:
        return index

    while n_words > 0 and index < len(text)-1:
        index += 1
        if text[index] is ' ':
            n_words -= 1

    return index


def preprocess(text):
    # remove new line symbol
    text = text.replace('\n', ' ')
    text = text.replace('_', ' ')
    text = text.replace('-', ' ')

    # replace multiple space with single space
    text = ' '.join(text.split())

    return text


def extract(txt_file, ann_file, file_name):
    with open(txt_file) as f:
        text = f.read()

    entities, relations, attributes, groups = get_entities_relations_attributes_groups(ann_file)

    '''
    entities: {'T1': Entity(id='T1', type='NoDisposition', span=((685, 691),), text='Reglan'), ..}
    attributes: {'A28': Attribute(id='A28', type='Temporality', target='E35', values=('Present',))}
    '''
    data = {'file':[], 'text':[], 'label':[],
    'Action':[], 'Negation':[], 'Temporality':[], 'Certainty':[], 'Actor':[]
    }
    for ent_id, entity in entities.items():
        start, end = entity.span[0]

        half_len = MAX_LEN/2
        start_window = adjust_prev_window(text, start, half_len)
        end_window = adjust_next_window(text, end, half_len)

        text_piece = text[start_window:start] + '<MED>' + text[end:end_window]

        text_piece = preprocess(text_piece)

        #print(entity, text_piece)
        data['file'].append(f'{file_name}_{ent_id}')
        data['text'].append(text_piece)
        data['label'].append(entity.type)


        #other labels: Action Negation Temporality Certainty Actor
        data['Action'].append(find_attr(attributes, 'Action', ent_id))
        data['Negation'].append(find_attr(attributes, 'Negation', ent_id))
        data['Temporality'].append(find_attr(attributes, 'Temporality', ent_id))
        data['Certainty'].append(find_attr(attributes, 'Certainty', ent_id))
        data['Actor'].append(find_attr(attributes, 'Actor', ent_id))

    return data


if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir =  sys.argv[2]
    out_file =  sys.argv[3]

    all_txt_files = glob.glob(f'{data_dir}/*txt')
    assert len(all_txt_files) != 0

    #print(all_txt_files)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for txt_file in all_txt_files:
        file_name = txt_file.replace(data_dir, '').replace('.txt', '')
        ann_file = txt_file.replace('txt', 'ann')
        data = extract(txt_file, ann_file, file_name)
        pd.DataFrame(data).to_csv(f'{out_dir}/{file_name}.csv', index=False)

    # merge csv files into one
    all_df = []
    for file_name in glob.glob(f'{out_dir}/*.csv'):
        file_id = file_name.replace(data_dir, '').replace('.csv', '')
        df = pd.read_csv(file_name)
        all_df.append(df)
    all_df = pd.concat(all_df, ignore_index=True)
    print('all:', len(all_df))
    all_df.to_csv(out_file, index=False)
