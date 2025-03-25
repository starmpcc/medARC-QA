#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: evaluate_from_api.py
Description: assess llms
License: [Type of license, e.g., "MIT License" or "GPLv3"]

"""

# Import libraries
import argparse
import ast 
import json
import os
import random
import re
import requests
import sys
import time


import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
from huggingface_hub import InferenceClient

from tqdm import tqdm
from datasets import load_dataset


OPENAI_API_KEY = ""
HF_API_KEY = ""
GOOGLE_API_KEY = ""
LAMBDA_API_KEY = ""
DEEPSEEK_API_KEY = ""
# curl -u <api_key>: https://cloud.lambdalabs.com/api/v1/instances

# medalpaca-13b-medarc
HF_INFERENCE_ENDPOINT = ""
ANTHROPIC_API_KEY = ""

def get_client(args):
    if args.model_name in ["microsoft/phi-4", "outputs/2025-02-26/03-03-48/", "Qwen/QwQ-32B", "Qwen/Qwen2.5-14B-Instruct","deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "OpenMeditron/Meditron3-Phi4-14B"]:
        openai.api_key = "test"
        client=OpenAI(
            api_key="test",
            base_url="http://localhost:8000/v1",
        )
    elif args.model_name in ["gpt-4", "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "o1-2024-12-17", "o3-mini"]:
        openai.api_key = OPENAI_API_KEY
        client = openai
    elif args.model_name in ["deepseek-ai/DeepSeek-R1"]:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
    elif args.model_name == "llama3.1-405b-instruct-fp8":
        # openai.api_key = LAMBDA_API_KEY
        # openai.base_url = "https://api.lambdalabs.com/v1"
        # client = openai
        client = OpenAI(
            api_key=LAMBDA_API_KEY,
            base_url="https://api.lambdalabs.com/v1",
        )
    elif args.model_name in ["Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct", "Llama-3.3-8B-Instruct", "Llama-3.3-70B-Instruct"]:
        client = InferenceClient(model="meta-llama/" + args.model_name, token=HF_API_KEY)
    elif args.model_name in ["Mistral-7B-Instruct-v0.3"]:
        client = InferenceClient(model="mistralai/" + args.model_name, token=HF_API_KEY)
    elif args.model_name in ["medalpaca-13b", "Meditron3-8B", "Meditron3-70B"]:
        client = InferenceClient(token=HF_API_KEY)
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash-8b"]:
        genai.configure(api_key=GOOGLE_API_KEY)
        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "max_output_tokens": 4000,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        client = genai.GenerativeModel(
            model_name=args.model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
        )


    else:
        client = None
        print("For other model API calls, please implement the client definition method yourself.")
    return client

def query(api_url, payload):
    headers = {
        "Accept" : "application/json",
        "Authorization": "Bearer hf_XXXXX",
        "Content-Type": "application/json" 
    }
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

def call_api(client, instruction, inputs):
    start = time.time()
    if args.model_name in ["microsoft/phi-4", "outputs/2025-02-26/03-03-48/", "Qwen/QwQ-32B", "Qwen/Qwen2.5-14B-Instruct","deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "OpenMeditron/Meditron3-Phi4-14B"]:
        message_text = [
            {"role": "user", "content": instruction + inputs},
        ]
        completion = client.chat.completions.create(
            model=args.model_name,
            messages=message_text,
            temperature=0,
            max_tokens=8000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["gpt-4", "gpt-4o", "gpt-4o-mini", "llama3.1-405b-instruct-fp8", "deepseek-ai/DeepSeek-R1"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
          temperature=0,
          max_tokens=8000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["o1-preview", "o1-mini", "o1-2024-12-17", "o3-mini"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        reasoning_effort= None
        if args.model_name == "o1-2024-12-17":
            if args.num_tokens is not None:
                num_tokens = args.num_tokens
        elif args.model_name == "o1-preview":
            num_tokens = 65536 # Max as of 10/14/2024, but not working in API
        elif args.model_name == "o1-mini":
            num_tokens = 65536 # Max as of 10/14/2024, but not working in API
        elif args.model_name == "o3-mini":
            num_tokens = 100000
            reasoning_effort = "high"
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
          # temperature=0, # Not supported as of 12/21/2024
          max_completion_tokens=num_tokens, # Not supported as of 10/14/2024
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          reasoning_effort=reasoning_effort,
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct", "Llama-3.3-8B-Instruct", "Llama-3.3-70B-Instruct"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
            model="meta-llama/" + args.model_name,
            messages=message_text,
            max_tokens=8000
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["Mistral-7B-Instruct-v0.3"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
            model="mistralai/" + args.model_name,
            temperature=0.0,
            messages=message_text,
            max_tokens=8000
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["medalpaca-13b", "Meditron3-8B", "Meditron3-70B",]:
        message_text = [{"role": "user", "content": instruction + inputs}]

        if 0:
            ### FYI Private model did not work w/completions (10/13/2024)
            # completion = client.chat.completions.create(
            #     # model="medalpaca/" + args.model_name,
            #     # model="TheBloke/medalpaca-13B-GGML",
            #     model="https://po55oh5wzv7ibwc2.us-east-1.aws.endpoints.huggingface.cloud",
            #     messages=message_text,
            #     max_tokens=8000
            # )
            completion = client.chat_completion(
                messages=message_text,
                model=HUGGINGFACE_API_URL,
            )
        elif args.model_name == "medalpaca-13b":
            completion = query(
                api_url=HF_INFERENCE_ENDPOINT,
                payload={
                "inputs": inputs,
                "parameters": {
                    "max_tokens": 8000
                }
            })
            print(completion)
            result = completion[0]['generated_text']
        elif args.model_name == "Meditron3-8B" or args.model_name == "Meditron3-70B":
            completion = query(
                api_url=HF_INFERENCE_ENDPOINT,
                payload={
                "inputs": inputs,
                "parameters": {
                    "max_tokens": 8000
                }
            })
            print(completion)
            result = completion[0]['generated_text']
        else:
            sys.exit("Huggingface error!")
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash-8b"]:
        chat_session = client.start_chat(
            history=[]
        )
        result = chat_session.send_message(instruction + inputs).text
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        message = client.messages.create(
            model=args.model_name,
            max_tokens=8000,
            system="",
            messages=[
                {"role": "user", "content": instruction + inputs}
            ],
            temperature=0.0,
            top_p=1,
        )
        result = message.content[0].text
    else:
        print("For other model API calls, please implement the request method yourself.")
        result = None
    print("cost time", time.time() - start)
    return result


def fix_options_column(example, debug=False):
    # Check if 'options' is a string and try to parse it as a list
    if isinstance(example.get('options'), str):
        if debug:
            print("Debugging question:", example.get('question_id', 'Unknown'))
            print("Question text:", example.get('question', 'No question text available'))
            print("Options string before parsing:", example['options'])
        
        # Replace various smart quotes
        cleaned_options = (
            example['options']
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
        )

        try:
            example['options'] = json.loads(cleaned_options)
        except json.JSONDecodeError as e:
            if debug:
                print("JSONDecodeError encountered!")
                print("Error message:", e)
                print("Problematic options string:", cleaned_options)
            raise e
    return example


def load_mmlu_pro():
    dev_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    dev_df = preprocess(dev_dataset["validation"])

    test_dataset = load_dataset("csv", data_files={"test":"data_medARCv1/test_medARCv1.csv"})
    test_dataset = test_dataset.map(fix_options_column)

    test_dataset.features = dev_dataset["test"].features
    test_df = preprocess(test_dataset["test"])

    return test_df, dev_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text):
    # Patterns for answer extraction
    patterns = [
        r"answer is \(?([A-J])\)?",                               # "answer is (D)" or "answer is D"
        r"[aA]nswer:\s*([A-J])",                                  # "Answer: D" or "answer: D"
        r"best course of action to perform immediately is"
        r" [\(\[]([A-J])[\)\]]",                                  # "best course... is (D)" or "[D]"
        r"best course of action.*[\(\[]([A-J])[\)\]]",            # "best course of action... (D)" or "[D]" in any sentence
        r"(?=.*\bbest\b)(?=.*\baction\b).*[\(\[]([A-J])[\)\]]",   # Sentence with "best" and "action" followed by (D) or [D]
    ]
    
    # Try each pattern sequentially
    for i, pattern in enumerate(patterns, start=1):
        match = re.search(pattern, text)
        if match:
            print(f"Pattern {i} matched: {pattern}")
            return match.group(1)
        else:
            print(f"{i} attempt failed\n" + text)
    
    # If no patterns matched
    return None


def single_request(client, single_question, cot_examples_dict, exist_result):
    exist = True
    q_id = single_question["question_id"]
    for each in exist_result:
        if q_id == each["question_id"] and single_question["question"] == each["question"]:
            pred = extract_answer(each["model_outputs"])
            return pred, each["model_outputs"], exist
    exist = False
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]

    question = process_age_strings(question) # Randomize ages to year +/- 2 weeks

    options = single_question["options"]
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    for each in cot_examples:
        prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    try:
        response = call_api(client, prompt, input_text)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None, exist
    pred = extract_answer(response)
    return pred, response, exist


def process_age_strings(text):
    """
    Finds age-related substrings in the input text, adds a random decimal
    between 0 and 0.04 to the age, and reformats the substring to "XX.XXX year-old".
    """
    age_pattern = re.compile(          # Define the regex pattern to match age substrings
        r'\b(\d+)'                     # Capture the numeric age
        r'(?:\s*[-\s]?)'               # Optional separator (space or hyphen)
        r'(year-old|year old|yo)\b',   # Capture the age descriptor
        re.IGNORECASE                  # Case-insensitive matching
    )
    
    def add_decimal(match):
        age = int(match.group(1))    # Extract the numeric age
        decimal = random.uniform(0, 0.04)
        new_age = age + decimal
        new_age_str = f"{new_age:.3f}"
        return f"{new_age_str} year-old"
    
    # Substitute all matched age substrings using the add_decimal function
    modified_text = age_pattern.sub(add_decimal, text)
    
    return modified_text


def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                        if not each["pred"]:
                            random.seed(12345)
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer_index"]:
                                category_record[category]["corr"] += 1
                            else:
                                category_record[category]["wrong"] += 1
                        elif each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if single["question_id"] == curr["question_id"] and single["question"] == curr["question"]:
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res


def evaluate(subjects, model_name, output_dir, args):
    client = get_client(args)
    test_df, dev_df = load_mmlu_pro()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    for subject in subjects:
        test_data = test_df[subject]

        if args.num_tokens is not None:
            num_tokens_str = '_' + str(args.num_tokens)
        else:
            num_tokens_str = ''

        output_res_path = os.path.join(output_dir, model_name.replace("/", "_") + num_tokens_str + '_' + subject + "_result.json")
        output_summary_path = os.path.join(output_dir, model_name.replace("/", "_") + num_tokens_str + '_' + subject + "_summary.json")
        res, category_record = update_result(output_res_path)

        for idx, each in enumerate(tqdm(test_data)):

            label = each["answer"]
            category = subject
            pred, response, exist = single_request(client, each, dev_df, res)
            if response is not None:
                res, category_record = update_result(output_res_path)
                if category not in category_record:
                    category_record[category] = {"corr": 0.0, "wrong": 0.0}
                each["pred"] = pred
                each["model_outputs"] = response
                merge_result(res, each)
                if pred is not None:
                    if pred == label:
                        category_record[category]["corr"] += 1
                    else:
                        category_record[category]["wrong"] += 1
                else:
                    category_record[category]["wrong"] += 1
                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)
                res, category_record = update_result(output_res_path)
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))


def save_summary(category_record, output_summary_path):
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4",
                        choices=["microsoft/phi-4", "outputs/2025-02-26/03-03-48/", "Qwen/QwQ-32B", "Qwen/Qwen2.5-14B-Instruct","deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "OpenMeditron/Meditron3-Phi4-14B", "deepseek-ai/DeepSeek-R1", "o3-mini", "gpt-4", "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "o1-2024-12-17", 
                                 "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct", 
                                 "Llama-3.3-8B-Instruct", "Llama-3.3-70B-Instruct", "llama3.1-405b-instruct-fp8",
                                 "Mistral-7B-Instruct-v0.3",
                                 "medalpaca-13b",
                                 "Meditron3-8B", "Meditron3-70B",
                                 "gemini-1.5-flash-latest",
                                 "gemini-1.5-pro-latest",
                                 "claude-3-opus-20240229",
                                 "gemini-1.5-flash-8b",
                                 "claude-3-sonnet-20240229"])
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    parser.add_argument("--uncertainty_quantification", "-uq", type=bool, default=False)
    parser.add_argument("--num_tokens", "-nt", type=int, default=None)

    assigned_subjects = []
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")

    if args.uncertainty_quantification:
        os.makedirs(os.path.join(args.output_dir, f"{args.model_name}"), exist_ok=True)

        # Create unique folder for each model run
        for n in range(10):
            curr_output_dir = os.path.join(args.output_dir, f"{args.model_name}", f"run_{n}")
            os.makedirs(curr_output_dir, exist_ok=True)
            evaluate(assigned_subjects, args.model_name, curr_output_dir, args)
            print("RUN DONE!")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        evaluate(assigned_subjects, args.model_name, args.output_dir, args)






