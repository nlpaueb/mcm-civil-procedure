export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/your_dir/model_weights/hf_cache
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
file_name="cot"

submission_name=$file_name
system="civil_procedure_system"
eval="evaluate_cot.py"
echo "Opening file: $file_name"

cat cot_prompt/${system}.txt | python $eval --seed 42 --experiment_name $submission_name --model_name model_weights/7B --peft_model peft_model_weights/$file_name