set -ex

dir_name=$1
model_name=$2
tokenizer_name=$3

output_path=output_${dir_name}
log_path=log_${dir_name}

if [ ! -d ${log_path} ]
then
	mkdir ${log_path}
fi

batch_size=32
grad=1
epoch=10
max_seq_len=128
save_steps=100

train_file='train.csv'
dev_file='dev.csv'
test_file='test.csv'

metric='f1_micro'
learning_rate='2e-5'

# experiments with 3 random initializations
for i in 42 62 82;
do
	data_path="n2c2_data"
	output_dir="${output_path}/model_${learning_rate}_${i}"

	if [ ! -d ${output_dir} ]
	then
		mkdir -p ${output_dir}

		python n2c2_code/model/run_classification.py \
			--seed ${i} \
			--model_name_or_path ${model_name} \
			--config_name ${model_name}  \
			--tokenizer_name ${tokenizer_name} \
			--task_name ${dir_name} \
			--data_dir  ${data_path} \
			--train_file  ${train_file} \
			--dev_file  ${dev_file} \
			--test_file  ${test_file} \
			--custom_metric ${metric} \
			--output_dir ${output_dir} \
			--per_device_train_batch_size ${batch_size} \
			--per_device_eval_batch_size ${batch_size} \
			--num_train_epochs ${epoch} \
			--overwrite_cache \
			--overwrite_output_dir \
			--save_steps ${save_steps} \
			--logging_steps 1 \
			--learning_rate ${learning_rate} \
			--gradient_accumulation_steps ${grad} \
			--max_seq_len ${max_seq_len} \
			--evaluate_during_training \
			--do_train --do_eval --do_predict
	fi
done
