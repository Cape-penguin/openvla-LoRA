import argparse
import json
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str, required=False, default='inference_results.json')
args = parser.parse_args()

# with open(args.json, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# print(f"Total sample length: {len(data)}")

data = []
with open(args.json, 'r', encoding='utf-8') as f:
    first_char = f.read(1)
    f.seek(0)  # 읽기 위치를 다시 처음으로 되돌림

    if first_char == '[':
        # 1. 일반적인 JSON 리스트 형식인 경우
        data = json.load(f)
        print(f"Loaded as Standard JSON. Total samples: {len(data)}")
    else:
        # 2. JSONL (줄바꿈 구분) 형식인 경우
        for line in f:
            line = line.strip()
            if line:  # 빈 줄이 아닌 경우에만 파싱
                data.append(json.loads(line))
        print(f"Loaded as JSONL. Total samples: {len(data)}")

def evaluate_tokens(gt_actions, pred_actions):

    gt_ids = np.round((np.array(gt_actions) + 1.0) / 2.0 * 255).astype(int)
    pred_ids = np.round((np.array(pred_actions) + 1.0) / 2.0 * 255).astype(int)
    
    token_diff = np.abs(gt_ids - pred_ids)
    
    mean_token_error = np.mean(token_diff) 
    exact_match = np.mean(token_diff == 0) 
    
    within_5_bins = np.mean(token_diff <= 5) 
    
    return {
        "mean_token_error": mean_token_error,
        "token_accuracy": exact_match,
        "near_accuracy_5": within_5_bins
    }

total_metrics = {
    "mean_token_error": [],
    "token_accuracy": [],
    "near_accuracy_5": []
}

for entry in data:
    gt_sample = entry['gt_action']
    pred_sample = entry['pred_action']
    
    metrics = evaluate_tokens(gt_sample, pred_sample)
    
    total_metrics["mean_token_error"].append(metrics["mean_token_error"])
    total_metrics["token_accuracy"].append(metrics["token_accuracy"])
    total_metrics["near_accuracy_5"].append(metrics["near_accuracy_5"])

# 2. 전체 평균 계산
final_results = {
    "avg_token_error": np.mean(total_metrics["mean_token_error"]),
    "avg_token_accuracy": np.mean(total_metrics["token_accuracy"]),
    "avg_near_accuracy_5": np.mean(total_metrics["near_accuracy_5"]),
    "total_samples": len(data)
}

print(f"Average token error: {final_results['avg_token_error']:.4f}")
print(f"Average token accuracy: {final_results['avg_token_accuracy'] * 100:.4f}%")
print(f"Average near accuracy (n=5): {final_results['avg_near_accuracy_5'] * 100:.4f}%")