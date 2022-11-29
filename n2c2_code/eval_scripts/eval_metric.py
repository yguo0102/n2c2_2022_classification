import sys
import pandas as pd
from transformers.data.metrics import simple_accuracy, pearson_and_spearman
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
import pdb

def read_csv(input_file, sep=','):
    df = pd.read_csv(open(input_file, 'r'), sep=sep, lineterminator='\n')
    return df


def evaluate(preds, labels, metric):
    res = None

    if 'INFORMATIVE' in labels:
        mapping = {'INFORMATIVE':1, 'UNINFORMATIVE':0}
        labels = [mapping[x] for x in labels]
        preds = [mapping[x] for x in preds]

    if 'ADE' in labels or 'NoADE' in labels:
        mapping = {'ADE':1, 'NoADE':0}
        labels = [mapping[x] for x in labels]
        preds = [mapping[x] for x in preds]

    if 'yes' in labels or 'no' in labels:
        mapping = {'yes':1, 'no':0}
        labels = [mapping[x] for x in labels]
        preds = [mapping[x] for x in preds]

    if metric == 'acc' :
        res = accuracy_score(preds, labels)
    elif metric == 'pearson' :
        res = pearson_and_spearman(preds, [float(x) for x in labels])['pearson']
    elif metric == 'f1_macro_weighted' :
        res = f1_score(y_true=labels, y_pred=preds, average='weighted')
    elif metric == 'f1_macro' :
        res = f1_score(y_true=labels, y_pred=preds, average='macro')
    elif metric == 'f1_micro' :
        res = f1_score(y_true=labels, y_pred=preds, average='micro')
    elif metric == 'pos_class_f1' :
        res = f1_score(y_true=labels, y_pred=preds)
    elif metric == 'precision' :
        res = precision_score(y_true=labels, y_pred=preds)
    elif metric == 'recall' :
        res = recall_score(y_true=labels, y_pred=preds)
    elif metric == 'neg_class_f1' :
        res = f1_score(y_true=labels, y_pred=preds, pos_label=0)
    elif metric == 'f1_pmabuse' :
        cls_repo = classification_report(y_true=labels, y_pred=preds, output_dict=True)
        res = cls_repo['0']['f1-score']
    elif metric == 'f1_report' :
        cls_repo = classification_report(y_true=labels, y_pred=preds, output_dict=True)
        print(classification_report(y_true=labels, y_pred=preds))
        res = '{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}'.format(cls_repo['1']['precision'], cls_repo['1']['recall'], cls_repo['1']['f1-score'], cls_repo['accuracy'])
    return res


if __name__ == '__main__':
    pred_file = sys.argv[1]
    label_file = sys.argv[2]
    metric = sys.argv[3]

    label_df = read_csv(label_file)
    labels = []
    for (i, (text, label)) in enumerate(zip(label_df['text'], label_df['label'])):
        if pd.isna(label): # skip NaN
            continue
        labels.append(label)

    pred_df = read_csv(pred_file, sep='\t')
    assert len(labels) == len(pred_df), 'gold:{}, pred:{}'.format(len(label_df), len(pred_df))

    preds = pred_df['prediction'].values.tolist()

    res = evaluate(preds, labels, metric)
    print('{}'.format(res))
