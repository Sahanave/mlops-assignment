import json, time, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

questions = [json.loads(l) for l in Path('evals/eval_set.jsonl').read_text().splitlines()][:10]

def fire(q):
    t0 = time.monotonic()
    r = requests.post('http://localhost:8000/v1/chat/completions', json={
        'model': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
        'messages': [{'role': 'user', 'content': q['question']}],
        'max_tokens': 300,
    })
    return time.monotonic() - t0

with ThreadPoolExecutor(max_workers=10) as ex:
    latencies = list(ex.map(fire, questions))

latencies.sort()
print(f'P50: {latencies[4]:.2f}s  P95: {latencies[-1]:.2f}s')