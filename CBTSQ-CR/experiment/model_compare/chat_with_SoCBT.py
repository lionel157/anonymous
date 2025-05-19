import os
import time
import json
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

client = OpenAI()

local_model_path = ".../SoCBT"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
local_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
)


dataset_file = "CBTSQ-CR/experiment/Test_Data/healme_result.json"
output_file = "CBTSQ-CR/experiment/result/SoCBT.json"


if not os.path.exists(output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)


with open(dataset_file, "r", encoding="utf-8") as f:
    samples = json.load(f)


with open(output_file, "r", encoding="utf-8") as f:
    cbt_en = json.load(f)

# therapist prompt templates
therapist_prompt_step_1 = " "

therapist_prompt_step_2 = " "

therapist_prompt_step_3 = " "

therapist_prompt_step_4 = " "


patient_prompt_template = " "


def chat_with_GPT(messages, retries=6, delay=2):
    attempt = 0
    while attempt < retries:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling  Chat: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying... ({attempt}/{retries})")
                time.sleep(delay)
            else:
                print("Max retries reached. Returning None.")
                return None


def chat_with_local_therapist(messages, retries=3, delay=2):
    attempt = 0
    while attempt < retries:
        try:
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"<|system|>\n{msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"<|user|>\n{msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"<|assistant|>\n{msg['content']}\n"
            prompt += "<|assistant|>\n"

            output = local_pipeline(prompt, max_new_tokens=200, temperature=0.7, do_sample=True)[0]["generated_text"]
            reply = output[len(prompt):].strip().split("\n")[0]
            return reply
        except Exception as e:
            print(f"Error calling local therapist model: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying... ({attempt}/{retries})")
                time.sleep(delay)
            else:
                print("Max retries reached. Returning None.")
                return None


MAX_HISTORY_ROUNDS = 3


for sample in samples:
    initial_patient_statement = sample["thought"]
    conversation = [{"role": "user", "content": initial_patient_statement}]
    initial_anchor = {"role": "user", "content": initial_patient_statement}

    example_id = len(cbt_en) + 1
    current_example = {"example_id": example_id, "dialogues": []}
    current_example["dialogues"].append({"speaker": "patient", "content": initial_patient_statement})

    stage = 1
    therapist_turns = 0

    while therapist_turns < 4:
        if stage == 1:
            therapist_prompt = therapist_prompt_step_1
            strategy = "Identify Cognitive Distortions + Empathetic Comfort"
        elif stage == 2:
            therapist_prompt = therapist_prompt_step_2
            strategy = "Challenging Negative Thoughts"
        elif stage == 3:
            therapist_prompt = therapist_prompt_step_3
            strategy = "Constructing Alternative Thoughts"
        elif stage == 4:
            therapist_prompt = therapist_prompt_step_4
            strategy = "Developing an Action Plan"

        window = conversation[-3:]
        input_context = [{"role": "system", "content": therapist_prompt}] + conversation


        therapist_response = chat_with_local_therapist(input_context)
        if therapist_response is None:
            print("Therapist response is None, ending conversation.")
            break

        print(f" Therapist (Step {stage}): {therapist_response}\n")
        conversation.append({"role": "assistant", "content": therapist_response})
        current_example["dialogues"].append({
            "speaker": "therapist",
            "content": therapist_response,
            "strategy": strategy
        })

        therapist_turns += 1

        if stage == 4:
            print(" Step 4 completed. Final therapist message delivered.\n")
            break


        patient_prompt = patient_prompt_template.format(therapist_response=therapist_response)
        input_context_patient = [{"role": "system", "content": patient_prompt}] + [initial_anchor] + conversation[-3:]


        patient_response = chat_with_GPT(input_context_patient)
        if patient_response is None:
            print("Patient response is None, ending conversation.")
            break

        print(f" Patient: {patient_response}\n")
        conversation.append({"role": "user", "content": patient_response})
        current_example["dialogues"].append({"speaker": "patient", "content": patient_response})

        if stage < 4:
            stage += 1


    cbt_en.append(current_example)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cbt_en, f, ensure_ascii=False, indent=2)

    print(f" Saved example {example_id} to file.\n")

print(f" All samples processed. Output file: {output_file}")
