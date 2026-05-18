import os
import re
import csv
import numpy as np

base_dir = '/home/kuonw/Documents/coast/unet-pytorch/img_out'
test_sets = ['20', '512']
models = ['舊資料訓練', '新資料接續訓練', '新資料接續訓練_100', '新資料混合訓練']

results = []

for ts in test_sets:
    for model in models:
        record_path = os.path.join(base_dir, ts, model, 'record')
        if not os.path.exists(record_path):
            print(f"File not found: {record_path}")
            continue
            
        with open(record_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        cm = []
        metrics = {}
        in_cm = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('Confusion Matrix:'):
                in_cm = True
                continue
            if in_cm:
                if '[' in line and ']' in line:
                    nums = [int(x) for x in re.findall(r'\d+', line)]
                    cm.append(nums)
                if len(cm) == 2:
                    in_cm = False
                    
            if line.startswith('mRecall:'): metrics['mRecall'] = float(line.split(':')[1])
            if line.startswith('mPrecision:'): metrics['mPrecision'] = float(line.split(':')[1])
            if line.startswith('mAcc:'): metrics['mAcc'] = float(line.split(':')[1])
            if line.startswith('mFallout:'): metrics['mFallout'] = float(line.split(':')[1])
            if line.startswith('mIoU:'): metrics['mIoU'] = float(line.split(':')[1])
            
        if len(cm) == 2:
            total = sum(sum(row) for row in cm)
            cm_perc = [[f"{val/total*100:.2f}%" for val in row] for row in cm]
            
            # Also provide the row-normalized percentage (true label percentage) if they meant that. 
            # I will just use overall percentage as it's straightforward, but let's label it clearly.
            
            results.append({
                'Test Set': ts,
                'Model': model,
                'CM_00 (TN)': cm_perc[0][0],
                'CM_01 (FP)': cm_perc[0][1],
                'CM_10 (FN)': cm_perc[1][0],
                'CM_11 (TP)': cm_perc[1][1],
                'mRecall': metrics.get('mRecall', ''),
                'mPrecision': metrics.get('mPrecision', ''),
                'mAcc': metrics.get('mAcc', ''),
                'mFallout': metrics.get('mFallout', ''),
                'mIoU': metrics.get('mIoU', '')
            })

csv_path = '/home/kuonw/Documents/coast/unet-pytorch/model_comparison.csv'
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    fieldnames = ['Test Set', 'Model', 'CM_00 (TN)', 'CM_01 (FP)', 'CM_10 (FN)', 'CM_11 (TP)', 'mRecall', 'mPrecision', 'mAcc', 'mFallout', 'mIoU']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
    
print(f"CSV written to {csv_path}")
