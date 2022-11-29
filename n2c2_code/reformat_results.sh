set -ex

for x in Action Actor Certainty Negation Temporality;
do
        python n2c2_code/merge_id_pred.py n2c2_data/test.csv pred_rb/${x}.txt pred_rb/${x}_fileid.txt

done

python n2c2_code/reformat_results_e2e.py 'n2c2_data/raw/test' 'n2c2_data/test.csv' 'pred_rb' 'pred_rb_final'
