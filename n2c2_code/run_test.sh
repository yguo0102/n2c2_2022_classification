set -ex

batch_size=48
train_file='train.csv'
dev_file='dev.csv'
test_file='test.csv'

data_path="n2c2_data/"

# the path of the optimal checkpoint
model_path="output_n2c2_roberta/model_2e-5_42/checkpoint-1100/"
tokenizer_name=${model_path}

# the location where the output file is saved
output_dir="./pred_rb"

max_seq_len=128
metric='f1_micro'

if [ ! -d $output_dir ];
then
	mkdir $output_dir
fi

python n2c2_code/model/run_classification.py \
	       --fp16 \
	       --max_seq_len ${max_seq_len} \
	       --model_name_or_path ${model_path} \
	       --config_name ${model_path}  \
	       --tokenizer_name ${tokenizer_name} \
	       --task_name n2c2 \
	       --data_dir  ${data_path} \
	       --train_file  ${train_file} \
	       --dev_file  ${dev_file} \
	       --test_file ${test_file} \
		--custom_metric ${metric} \
	       --output_dir ${output_dir} \
	       --per_device_train_batch_size ${batch_size} \
	       --per_device_eval_batch_size ${batch_size} \
	       --overwrite_output_dir \
	       --overwrite_cache \
	       --logging_steps 1 \
	       --do_predict
	       date
