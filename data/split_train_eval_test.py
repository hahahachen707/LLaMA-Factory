import json
import random

def split_json_file(input_file, train_file, eval_file, test_file, train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1, seed=42):
    # 读取原始json
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 打乱顺序
    random.seed(seed)
    random.shuffle(data)
    
    total = len(data)
    train_end = int(total * train_ratio)
    eval_end = train_end + int(total * eval_ratio)
    
    train_data = data[:train_end]
    eval_data = data[train_end:eval_end]
    test_data = data[eval_end:]
    
    # 保存为三个json文件
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=4)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

# 使用示例，假设要划分的是qa_pairs_5_alpaca.json
if __name__ == '__main__':
    input_path = 'qa_pairs_5_alpaca.json'
    split_json_file(
        input_file=input_path,
        train_file='qa_pairs_5_alpaca_train.json',
        eval_file='qa_pairs_5_alpaca_eval.json',
        test_file='qa_pairs_5_alpaca_test.json'
    )
