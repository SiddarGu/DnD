import json

with open('test.jsonl') as f:
    data = [json.loads(line) for line in f]

for d in data:
    input = d['input']
    skill = d['skill']
    target = d['target']
    intent = input.split('</s>')[-1].strip()
    input_without_intent = input.replace(intent, '').replace('</s>', '').strip()

    # write to jsonl
    with open('test_cleaned.jsonl', 'a') as f:
        json.dump({
            'input': input_without_intent,
            'skill': skill,
            'target': target,
            'intent': intent
        }, f)
        f.write('\n')