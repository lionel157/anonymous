import os
import json
from openai import OpenAI
from tqdm import tqdm

# Initialize client
client = OpenAI()

# Load baseline samples
with open(r"CBTSQ-CR/Data/SoCBTtalk_2000.json", "r",
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
output_file = r"...\SoCBT_scores.json"
summary_file = r"...\SoCBT_dimension_avg.json"

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


dimension_scores = {
    "Emotional Support": [],
    "Dialogue Naturalness": [],
    "Restructuring Effectiveness": [],
    "Therapist Adaptability": [],
    "Guidance Quality": []
}

for idx, base_data in tqdm(enumerate(baseline_samples), total=len(baseline_samples)):
    example_id = base_data.get("example_id", idx)

    if example_id in processed_ids:
        continue

    dialogue = format_dialogue(base_data["dialogues"], {"patient": "Patient", "therapist": "Therapist"})

    prompt = f" "

    try:
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
        score = json.loads(response_text)


        values = [
            score["Emotional Support"],
            score["Dialogue Naturalness"],
            score["Restructuring Effectiveness"],
            score["Therapist Adaptability"],
            score["Guidance Quality"]
        ]
        score["average_score"] = round(sum(values) / len(values), 2)


        dimension_scores["Emotional Support"].append(score["Emotional Support"])
        dimension_scores["Dialogue Naturalness"].append(score["Dialogue Naturalness"])
        dimension_scores["Restructuring Effectiveness"].append(score["Restructuring Effectiveness"])
        dimension_scores["Therapist Adaptability"].append(score["Therapist Adaptability"])
        dimension_scores["Guidance Quality"].append(score["Guidance Quality"])

    except Exception as e:
        score = {"error": str(e), "raw": response_text if 'response_text' in locals() else None}


    result = {
        "example_id": example_id,
        "dialogue": dialogue,
        "evaluation": score
    }

    all_scores.append(result)
    processed_ids.add(example_id)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)


dimension_avg = {
    "emotional_supportiveness_avg": round(
        sum(dimension_scores["Emotional Support"]) / len(dimension_scores["Emotional Support"]), 2),
    "dialogue_naturalness_avg": round(
        sum(dimension_scores["Dialogue Naturalness"]) / len(dimension_scores["Dialogue Naturalness"]), 2),
    "cognitive_restructuring_avg": round(
        sum(dimension_scores["Restructuring Effectiveness"]) / len(dimension_scores["Restructuring Effectiveness"]), 2),
    "therapist_appropriateness_avg": round(
        sum(dimension_scores["Therapist Adaptability"]) / len(dimension_scores["Therapist Adaptability"]), 2),
    "guidance_effectiveness_avg": round(
        sum(dimension_scores["Guidance Quality"]) / len(dimension_scores["Guidance Quality"]), 2),
    "total_samples": len(all_scores)
}


with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(dimension_avg, f, indent=2, ensure_ascii=False)


print(json.dumps(dimension_avg, indent=2, ensure_ascii=False))
