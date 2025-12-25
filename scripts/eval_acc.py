import json

# 定义文件路径
file = '/apdcephfs_cq12/share_1150325/hahahachen/work/LLaMA-Factory/saves/qwen3vl-4b/lora/test/generated_predictions.jsonl'

def extract_task_completion(json_str_wrapped):
    """
    从包含JSON的字符串中提取task_completion字段
    """
    if not json_str_wrapped or not isinstance(json_str_wrapped, str):
        return None
        
    # 尝试找到JSON部分
    # JSON通常在 {...} 中
    # 寻找第一个 { 和最后一个 }
    
    try:
        start = json_str_wrapped.find('{')
        end = json_str_wrapped.rfind('}')
        
        if start != -1 and end != -1:
            potential_json = json_str_wrapped[start:end+1]
            data = json.loads(potential_json)
            return data.get('task_completion')
    except Exception as e:
        pass
        
    return None

def process_file(file_path):
    preds = []
    gts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                # 每一行本身是一个JSON对象
                row = json.loads(line)
                
                # 获取 predict 和 label 内容
                predict_content = row.get('predict', '')
                label_content = row.get('label', '')
                
                # 提取 task_completion
                pred = extract_task_completion(predict_content)
                gt = extract_task_completion(label_content)
                
                preds.append(pred)
                gts.append(gt)
                
            except json.JSONDecodeError:
                print(f"Line {i+1} is not valid JSON")
                continue
                
    return preds, gts

def calc_acc(preds, gts):
    assert len(preds) == len(gts), f"数量对不上: preds({len(preds)}), gts({len(gts)})"
    
    match_count = 0
    
    for p, g in zip(preds, gts):
        # 标准化比较 (转小写, 去空格)
        p_val = str(p).strip().lower() if p is not None else "none_pred"
        g_val = str(g).strip().lower() if g is not None else "none_gt"
        
        if p_val == g_val:
            match_count += 1
        
    acc = match_count / len(preds) if preds else 0.0
    print(f"Accuracy: {acc:.4f} ({match_count}/{len(preds)})")
    return acc

if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    preds, gts = process_file(file)
    calc_acc(preds, gts)
