import os
import json
from openai import OpenAI
from tqdm import tqdm
import time

# Initialize client
client = OpenAI()

# Load baseline samples
with open(r"CBTSQ-CR\experiment\result\SoCBT_result.json", "r",
          encoding="utf-8") as f:
    baseline_samples = json.load(f)

def format_dialogue(dialogue, speaker_map):
    text = ""
    for turn in dialogue:
        speaker = speaker_map.get(turn["speaker"].lower(), turn["speaker"])
        content = turn.get("content") or turn.get("text")
        text += f"{speaker}: {content}\n"
    return text

# Output file path
output_file = r" "
summary_file = r" "

# Initialize output files
if not os.path.exists(output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)

# Load existing results
try:
    with open(output_file, "r", encoding="utf-8") as f:
        all_scores = json.load(f)
    processed_ids = {item["example_id"] for item in all_scores}
except:
    all_scores = []
    processed_ids = set()

# Initialize dimension scores collector
dimension_scores = {
    "emotional_supportiveness": [],
    "dialogue_naturalness": [],
    "cognitive_restructuring": [],
    "therapist_appropriateness": [],
    "guidance_effectiveness": []
}

# Retry mechanism
def api_call_with_retry(func, max_retries=6, delay=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e), "attempts": attempt + 1}
            time.sleep(delay * (2 ** attempt))  # Exponential backoff
            print(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")

for idx, base_data in tqdm(enumerate(baseline_samples), total=len(baseline_samples)):
    example_id = base_data.get("example_id", idx)

    if example_id in processed_ids:
        continue

    dialogue = format_dialogue(base_data["dialogues"], {"patient": "Patient", "therapist": "Therapist"})

    prompt = f" "

    def call_api():
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating psychotherapy dialogue."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        response_text = response.choices[0].message.content
        return json.loads(response_text)

    result = api_call_with_retry(call_api)

    if "error" not in result:
        score = result
        # Calculate average score for the sample
        values = [
            score["Emotional Support"],
            score["Dialogue Naturalness"],
            score["Restructuring Effectiveness"],
            score["Therapist Adaptability"],
            score["Guidance Quality"]
        ]
        score["average_score"] = round(sum(values) / len(values), 2)

        # Collect dimension scores
        dimension_scores["Emotional Support"].append(score["Emotional Support"])
        dimension_scores["Dialogue Naturalness"].append(score["Dialogue Naturalness"])
        dimension_scores["Restructuring Effectiveness"].append(score["Restructuring Effectiveness"])
        dimension_scores["Therapist Adaptability"].append(score["Therapist Adaptability"])
        dimension_scores["Guidance Quality"].append(score["Guidance Quality"])
    else:
        score = result

    # Save individual sample result
    result = {
        "example_id": example_id,
        "dialogue": dialogue,
        "evaluation": score
    }

    all_scores.append(result)
    processed_ids.add(example_id)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)

# Calculate dimension averages
dimension_avg = {
    "emotional_supportiveness_avg": round(
        sum(dimension_scores["Emotional Support"]) / len(dimension_scores["Emotional Support"]), 2) if dimension_scores["Emotional Support"] else 0,
    "dialogue_naturalness_avg": round(
        sum(dimension_scores["Dialogue Naturalness"]) / len(dimension_scores["Dialogue Naturalness"]), 2) if dimension_scores["Dialogue Naturalness"] else 0,
    "cognitive_restructuring_avg": round(
        sum(dimension_scores["Restructuring Effectiveness"]) / len(dimension_scores["Restructuring Effectiveness"]), 2) if dimension_scores["Restructuring Effectiveness"] else 0,
    "therapist_appropriateness_avg": round(
        sum(dimension_scores["Therapist Adaptability"]) / len(dimension_scores["Therapist Adaptability"]), 2) if dimension_scores["Therapist Adaptability"] else 0,
    "guidance_effectiveness_avg": round(
        sum(dimension_scores["Guidance Quality"]) / len(dimension_scores["Guidance Quality"]), 2) if dimension_scores["Guidance Quality"] else 0,
    "total_samples": len(all_scores)
}

# Save dimension averages
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(dimension_avg, f, indent=2, ensure_ascii=False)


print(json.dumps(dimension_avg, indent=2, ensure_ascii=False))
