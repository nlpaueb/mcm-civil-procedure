# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model

from accelerate.utils import is_xpu_available

import datasets
from tqdm import tqdm

import csv

def get_input_text(system_prompt, sample):
    return f"{system_prompt}\nIntroduction:\n{sample['explanation']}\nQuestion:\n{sample['question']}\nAnswer:\n{sample['answer']}\nExplanation: "

# def get_input_text_sum(system_prompt, sample):
#     # return f"{system_prompt}\nQuestion:\n{sample['question']}\nAnswer:\n{sample['answer']}\nLabel:"
#     return f"{system_prompt}\nIntroduction:\n{sample['intro_summaries']}\nQuestion:\n{sample['question']}\nAnswer:\n{sample['answer']}\nExplanation: "


def main(
    model_name,
    system_file: str=None,
    split: str='test',
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool=False,
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    base_dir = "./results/submissions"
    filename = "semeval_dev"
    import os

    def file_with_unique_name(base_dir, filename, extension='csv'):
        index = 0
        file_path = f"{base_dir}/{filename}_{index}.{extension}"
        while os.path.exists(file_path):
            index += 1
            file_path = f"{base_dir}/{filename}_{index}.{extension}"
        return file_path
    
    filepath = file_with_unique_name(base_dir, filename)

    dataset = datasets.load_dataset('csv', data_files = {
        'test':'./data/dev.csv',
        })
    dataset = dataset[split]
    print(f'Dataset length: {len(dataset)}')

    if system_file is not None:
        assert os.path.exists(
            system_file
        ), f"Provided Prompt file does not exist {system_file}"
        with open(system_file, "r") as f:
            system_prompt = "\n".join(f.readlines())
    elif not sys.stdin.isatty():
        system_prompt = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
    
    # Setup model
        ##############
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Here we assemble the input
    # sample = {
    #     'explanation':"State law is not static. It may change while a federal diversity case is being litigated, before it is litigated, after it is litigated, or while it is on appeal. If it changes during the litigation, should the federal court conform to the state law as it existed when the case arose, when it was filed, when it was tried, or as of some other point in the litigation? Here’s a question that illustrates the problem. The basic premises of Erie should help you to choose the best answer.",
    #     'question': "7. Comedy of errors. Laurel and Hardy are injured when they are traveling in Fields’s car in the State of Emporia and Fields collides with Gleason. Laurel and Hardy bring a diversity action against Fields and Gleason in federal court. Under Emporia case law, it is unclear whether, if Gleason and Fields were both negligent, they would be liable for their percentages of fault (several liability) or each liable for the plaintiff’s full damages (joint and several liability). The federal district judge, making her best guess at the law of Emporia on the liability of joint tortfeasors, concludes that the Emporia Supreme Court would apply joint and several liability. Consequently, after the jury finds both Fields and Gleason negligent, she holds each defendant liable for Laurel and Hardy’s full damages, rather than for reduced damages based on each defendant’s percentage of negligence. Fields appeals her ruling to the federal Court of Appeals. Gleason does not appeal. While Fields’s appeal is pending, the Emporia Supreme Court holds, in an unrelated case, that several liability is the proper rule in joint tortfeasor cases. Fields’s federal appeal is heard after the Emporia Supreme Court’s decision comes down. Fields argues that the Court of Appeals should apply several liability to his case, even though the trial judge had applied joint and several liability. Gleason, the other defendant, moves for relief from the judgment against him in the federal district court, on the ground that the judgment was based on a mistake concerning the applicable law. The federal Court of Appeals will",
    #     'answer':"reverse the decision in Fields’s case, and remand it for entry of a new judgment based on several liability. The federal district court will probably deny Gleason’s motion for relief from judgment.",
    # }
    correct = 0
    wrong=0
    failed_memory=0
    F=[]
    # total = len(dataset)
    answer_list = []
    mistakes_list = []
    for sample in tqdm(dataset):
        label = sample['label']
        idx = sample['idx']
        # large_samples = []# [5, 8, 12, 13, 16, 17, 18, 19, 20, 21, 23, 24, 29, 30, 31, 33, 34, 35, 36, 54, 67]
        try:
            # if idx not in large_samples:
            input_text = get_input_text(system_prompt, sample)
            batch = tokenizer(input_text, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
            # call inference script
            outputs = inference(
                model,
                batch,
                max_new_tokens,
                do_sample,
                min_length,
                use_cache,
                top_p,
                temperature,
                top_k,
                repetition_penalty,
                length_penalty,
                **kwargs
            )
            
        except torch.cuda.OutOfMemoryError:
            failed_memory+=1
            F.append(sample['idx'])
            answer_list.append((idx, None))
            continue

        # return Output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"{output_text[output_text.find('Label: '):]}")
        # print(output_text)
        # print(output_text[output_text.find('Label: ')+len('Label: '):])

        # extract answer
        label_idx = output_text.lower().find('label:')
        output_cot = output_text[:label_idx]
        output_label = output_text[label_idx+len('label:'):].lower()
        if output_label.count('true')==1:
            pred = 1
        else:
            pred = 0
        
        answer_list.append((idx, pred))
        print("Label: ", output_label,", Prediction: ", pred)
        print("Actual Label: ", str(bool(label) ), end=', ')
        
        if pred==bool(label):
            correct+=1
            print('correct')
        else:
            # print(output_label, )
            wrong+=1
            mistakes_list.append(idx)
            print('wrong')
        # print(bool(label))
        # torch.cuda.empty_cache()
    print(f'Correct: {correct}, Total: {correct+wrong}, Accuracy: {100*correct/(correct+wrong):.2f}')
    print(f'mistakes list: {mistakes_list}')
    print(f'failed_memory: {failed_memory}')
    print(f' list of failed_memory ids: {F}')
    print(answer_list)

    print(output_cot)

    for idx, pred in answer_list:
        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['idx', 'pred'])
            # Writing data
            writer.writerow([idx, pred])




def inference(
    model,
    batch,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    **kwargs
):
    if is_xpu_available():
        batch = {k: v.to("xpu") for k, v in batch.items()}
    else:
        batch = {k: v.to("cuda") for k, v in batch.items()}

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs 
        )
    e2e_inference_time = (time.perf_counter()-start)*1000
    # print(f"the inference time is {e2e_inference_time} ms")

    return outputs

if __name__ == "__main__":
    fire.Fire(main)
