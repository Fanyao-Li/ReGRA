import re
import os
import json
import copy 
import torch
from tqdm import tqdm
from datetime import datetime


MATH_DATASETS = ['AddSub', 'MultiArith', 'SingleEq', 'SVAMP', 'gsm8k', 'AQuA']
COMMONSENSE_DATASETS = ["openbookqa", "ARC-Challenge", "winogrande", "piqa", "social_i_qa", "ARC-Easy", "boolq", "hellaswag"]


## evaluate functions
def extract_answer(dataset, sentence: str) -> float:
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

def extract_answer_number(dataset, sentence: str) -> float:
    dataset = dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def extract_answer_letter(dataset, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[-1]
    else:
        return ''

def generate_predict_from_answer(dataset, output, label, miss):
    flag = False
    if dataset in COMMONSENSE_DATASETS:
        predict = extract_answer(dataset, output)
        if label == predict:
            flag = True
    elif dataset == "AQuA":
        predict = extract_answer_letter(dataset, output)
        if label == predict:
            flag = True
    else:
        if isinstance(label, str):
            label = float(label)
        predict = extract_answer_number(dataset, output)
        if abs(label - predict) <= miss:
            flag = True
    return predict, flag

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches

def generate_prompt_eval(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {instruction}

            ### Response:
            """  # noqa: E501

def evaluate(model, tokenizer, instructions, max_new_tokens=256):
    prompts = [generate_prompt_eval(instruction) for instruction in instructions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(input_ids,
            attention_mask=inputs['attention_mask'].to(model.device),
            do_sample=False,
            # generation_config=GenerationConfig(num_beams=4),      # can be used for better accuracy, but slower
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    outputs = [o.split("### Response:")[-1].strip() for o in outputs]
    return outputs

def write_eval_result(output_dir, dataset_name, output_data, correct, total, start_time):
    end_time = datetime.now()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"total time: {int(hours)} hours, {int(minutes)} minutes")

    os.makedirs(output_dir, exist_ok=True)
    save_file = f'{output_dir}/{dataset_name}.json'
    with open(save_file, 'w+') as f:
        json.dump(output_data, f, indent=4)

    acc_file = f'{output_dir}/../accuracy.json'
    if os.path.exists(acc_file):
        acc = json.load(open(acc_file, 'r'))
    else:
        acc = {"all": {"avg_accuracy": 0}}
    acc[dataset_name] = {"correct": correct, 
                    "total": total, 
                    "accuracy": correct/total, 
                    "time": f"{int(hours)}:{int(minutes)}",
                    }
    dataset_list = []
    accuracy_list = []
    for k, v in acc.items():
        if k != "all":
            dataset_list.append(k)
            accuracy_list.append(v["accuracy"])
    acc['all'] = {"avg_accuracy": sum(accuracy_list)/len(accuracy_list), 
                  "accuracy_list": accuracy_list,
                  "dataset_list": dataset_list, 
                  "time": f"{end_time.strftime('%Y-%m-%d %H:%M:%S')}",
                   }
    with open(acc_file, 'w+') as f:
        json.dump(acc, f, indent=4)

def evaluate_dataset(model, tokenizer, datamode,dataset_name, batch_size, output_dir):
    start_time = datetime.now()
    ## dataset
    dataset = json.load(open(f'dataset/{datamode}/{dataset_name}/test.json', 'r'))
    batches = create_batch(dataset, batch_size)
    ## evaluate
    total = len(batches)
    correct, current, output_data = 0, 0, []
    miss = 0.001
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]
        outputs = evaluate(model, tokenizer, instructions)
        for data, output in zip(batch, outputs):
            label = data.get('answer')
            predict, flag = generate_predict_from_answer(dataset_name, output, label, miss)
            if flag == True:
                correct += 1
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            print(new_data)
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        pbar.update(1)
    pbar.close()
    print('\n test finished')

    write_eval_result(output_dir, dataset_name, output_data, correct, current, start_time)
    
    return correct / current
