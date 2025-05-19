import os
import json
import time
from openai import OpenAI


client = OpenAI()


dataset_file = r"CBTSQ-CR/experiment/Test_Data/healme_result.json"
output_file = r"CBTSQ-CR/experiment/result/gpt3_5_result.json"


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


def chat_with_gpt(messages, max_retries=6, initial_delay=3):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f" Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f" Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(" Maximum retry attempts reached. Skipping this message.")
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
            strategy = "Identifying Distorted Cognition"
        elif stage == 2:
            therapist_prompt = therapist_prompt_step_2
            strategy = "Challenge Negative Thoughts"
        elif stage == 3:
            therapist_prompt = therapist_prompt_step_3
            strategy = "Constructing Alternative Thoughts"
        elif stage == 4:
            therapist_prompt = therapist_prompt_step_4
            strategy = "Summarizing Insights and Offering Optional Suggestions"

        window = conversation[-3:]
        input_context = [{"role": "system", "content": therapist_prompt}] + conversation

        therapist_response = chat_with_gpt(input_context)
        if therapist_response is None:
            print("Therapist response is None, ending conversation.")
            break

        print(f"Therapist (Step {stage}): {therapist_response}\n")
        conversation.append({"role": "assistant", "content": therapist_response})
        current_example["dialogues"].append({"speaker": "therapist", "content": therapist_response, "strategy": strategy})

        therapist_turns += 1

        if stage == 4:
            print(" Step 4 completed. Final therapist message delivered.\n")
            break


        patient_prompt = patient_prompt_template.format(therapist_response=therapist_response)
        input_context_patient = [{"role": "system", "content": patient_prompt}] + [initial_anchor] + conversation[-3:]

        patient_response = chat_with_gpt(input_context_patient)
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
