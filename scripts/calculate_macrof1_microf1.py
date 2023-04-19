import pandas as pd
import re 

from pathlib import Path 

dicti_gr = {}
dicti_pred = {}
dicti_correct = {}
to_check = []

dicti_f1 = {}
dicti_f1_micro = {}

df = pd.read_csv('/home/shivani/work/vilt/output_truth27.tsv', sep='\t')
for j in range(len(df)):
    gt, pred = df.iloc[j]
    for i in eval(gt):
        if i not in dicti_gr:
            dicti_gr[i] = 1
        else:
            dicti_gr[i] += 1
    
    for i in eval(pred):
        if i not in dicti_pred:
            dicti_pred[i] = 1
        else:
            dicti_pred[i] += 1

for j in range(len(df)):
    gt, pred = df.iloc[j]
    gt = eval(gt)
    pred = eval(pred)
    correct = set(gt) & set(pred)
    for i in correct:
        if i not in dicti_correct:
            dicti_correct[i] = 1
        else:
            dicti_correct[i] += 1

weight = 0
for i in dicti_gr:
    if i not in dicti_pred or i not in dicti_correct:
        to_check.append(i)
        pr = 0
    else:
        pr = dicti_correct[i]/dicti_pred[i]

    if i not in dicti_correct:
        re = 0
    else:
        re = dicti_correct[i]/dicti_gr[i]

    if pr == 0 or re == 0:
        f1 = 0
    else:
        f1 = (2* pr * re)/(pr + re)
    dicti_f1[i] = f1
    dicti_f1_micro[i] = dicti_gr[i] * f1
    weight += dicti_gr[i]

sum_ = 0
sum_micro = 0
for i in dicti_f1:
    sum_ += dicti_f1[i]

for i in dicti_f1_micro:
    sum_micro += dicti_f1_micro[i]

print("Macro F1 Score: ", sum_/len(dicti_gr))
print("Micro F1: ", sum_micro/weight )
print(to_check)
