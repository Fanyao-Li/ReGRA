from datasets import load_dataset


## finetune functions
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501

def tokenize(prompt, tokenizer, max_length, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_length
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point, tokenizer, max_length, train_on_inputs):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt, tokenizer, max_length, add_eos_token=True)
    if not train_on_inputs:
        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, tokenizer, max_length, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt

def get_data(data_path, val_set_size, tokenizer, max_length=256, train_on_inputs=False, seed=42):
    data = load_dataset("json", data_files=data_path)
    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=int(seed))
        train_data = train_val["train"].shuffle(seed=int(seed)).map(lambda x: generate_and_tokenize_prompt(x, tokenizer, max_length, train_on_inputs))
        val_data = train_val["test"].shuffle(seed=int(seed)).map(lambda x: generate_and_tokenize_prompt(x, tokenizer, max_length, train_on_inputs))
    else:
        train_data = data["train"].shuffle(seed=int(seed)).map(lambda x: generate_and_tokenize_prompt(x, tokenizer, max_length, train_on_inputs))
        val_data = None
    return train_data, val_data
