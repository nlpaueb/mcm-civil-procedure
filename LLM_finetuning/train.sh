export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dir/model_weights/hf_cache
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
file_name="cot"
dataset_name=$file_name
echo "Opening file: $file_name"
echo

python quickstart.py --quantization --batch_size 2 --gradient_accumulation_steps 2 --bf16  --output_dir peft_model_weights/${file_name} --custom_data_file custom_datasets/${dataset_name}/custom_dataset.py 