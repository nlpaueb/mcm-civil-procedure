import datasets


def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset('csv', data_files = {
        'train': './data/train_cot.csv',
        'validation': './data/train_cot.csv',
        # 'test':'./data/dev.csv',
        })
    dataset = dataset[split]

    instructions = "You are an expert lawyer in the domain of U.S. civil procedure. You are given an introduction to a legal case, a question about that case and an answer to that question. You have to decide if the answer is correct or not. First explain why the answer is correct or not and then output the label of the answer with one word either True or False."
    prompt = (
        f"Instructions: {instructions}\nIntroduction: {{introduction}}\nQuestion: {{question}}\nAnswer: {{answer}}\nExplanation: "
    )

    label_text = (
        f"{{cot}}\nLabel: {{label}}."
    )
    
    label_map = {
        1: "True",
        0: "False"
    }
    def apply_prompt_template(sample):
        # print(label_map[sample['label']])
        return {
            "prompt": prompt.format(introduction=sample['explanation'], question=sample['question'], answer=sample['answer']),
            "label": label_text.format(cot=sample['cot'],label=label_map[sample['label']])
        }

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        label = tokenizer.encode(sample['label'] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + label,
            "attention_mask" : [1] * (len(prompt) + len(label)),
            "labels": [-100] * len(prompt) + label,
            }
        return sample

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
