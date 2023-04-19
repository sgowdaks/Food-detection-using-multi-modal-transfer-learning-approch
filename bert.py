#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import csv
import json
import torch
import tensorflow as tf
import random
import torch
import torch.nn as nn
import time
import datetime
import transformers
import argparse

from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, AutoModelForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-03
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Trainer:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def create_data(self):
        column_names =  ['diaries', 'usda_id', 'image_id']
        df_usda = pd.DataFrame(columns = column_names)

        dicti = {}
        
        with open(self.data_dir, 'r') as input:
            csv_reader =  csv.reader(input, delimiter='\t')
            for line in csv_reader:
                # if not isinstance(line, list):
                if line[1] == "usda_id":
                    continue
                categories = eval(line[1])
                for category in categories:
                    if category in dicti:
                        dicti[category] += 1
                    else:
                        dicti[category] = 1
        
            sorted_dict = dict(sorted(dicti.items(), key=lambda item: item[1]))
            count = 0

            all_the_categories = []
            for val in sorted_dict:
                if sorted_dict[val] < 1:
                    count += 1
                else:
                    if val not in all_the_categories:
                        all_the_categories.append(val)

        with open(self.data_dir, 'r') as input:
            csv_reader =  csv.reader(input, delimiter='\t')
            for line in csv_reader:
                if line[1] == "usda_id":
                    continue
                categories = eval(line[1])
                usda_ids = []
                for category in categories:
                    if category in sorted_dict:
                        if sorted_dict[category] >= 1 :
                            usda_ids.append(category)
                        else:
                            usda_ids.append("UNK")                        
                dicti = pd.DataFrame({'diaries': line[0], 'usda_id': str(usda_ids), 'image_id': line[2]}, index=[0])
                df_usda = pd.concat([df_usda, dicti], ignore_index = True, axis = 0) 
        return df_usda[['diaries', 'usda_id']], all_the_categories

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.diaries
        self.targets = self.data.usda_id
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text.iloc[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[index], dtype=torch.float)
        }

class BERTClass(torch.nn.Module):
    def __init__(self, id2label):
        super(BERTClass, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.head = nn.Sequential(
            torch.nn.Linear(768, 1000), 
            nn.GELU(), 
            torch.nn.Linear(1000, len(id2label)))
    
    def forward(self, ids, mask, token_type_ids):
        seq_rep , _ = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)
        # print(output_1.shape)
        sen_rep = seq_rep[:, 0, :]
        cls_rep = self.head(sen_rep)

        return cls_rep
    
def train_loop(model, optimizer, training_loader, epoch, testing_loader, scheduler, loss_fn):
    model.train()
    optimizer.zero_grad()
    with tqdm(training_loader, desc=f"Train Ep={epoch}", dynamic_ncols=True) as databar:
        total_loss = 0.0
        n_updates = 0
        for data in databar:
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            loss = loss.sum(dim=1).mean() 
            total_loss += loss.item()
            n_updates += 1
            databar.set_postfix_str(f'itLoss: {loss.item():4f} avgLoss={total_loss/n_updates:.4f}')   
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    if True:
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for data in tqdm(testing_loader, desc=f"Validation Ep={epoch}"):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        outputs_, targets = fin_outputs, fin_targets
        print(max(outputs_[0]))
        outputs = np.array(outputs_) >= 0.01
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy:.4f}")
        print(f"F1 Score (Micro) = {f1_score_micro:.4f}")
        print(f"F1 Score (Macro) = {f1_score_macro:.4f}")
    scheduler.step()
    
def evaluate(MODEL_PATH, model, validation_loader, id2label, validation_dataset):
    with open(MODEL_PATH / 'output_truth19.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['ground truth', 'prediction'])
        fin_targets=[]
        fin_outputs=[]

        # state_dicti = torch.load(MODEL_PATH / 'chkpt-exp-01400020.pt' )
        # model.load_state_dict(state_dicti['model_state_dict'])
        # model = model.to(device)
        # model.eval()

        for i, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            logits = torch.sigmoid(outputs).cpu().detach().numpy()[0]
            # print(validation_dataset[i])

            l = logits
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    
            predictions = np.zeros(len(id2label))    
            predictions[np.where(logits > 0.1)] = 1

            predictions = predictions.flatten()
            predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
            ground_truth = [id2label[idx] for idx, label in enumerate(validation_dataset.iloc[i]['usda_id']) if label == 1.0]
            print(predicted_labels, ground_truth)
            tsv_writer.writerow([ground_truth, predicted_labels])

        outputs_, targets = fin_outputs, fin_targets
        outputs = np.array(outputs_) >= 0.01
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy:.4f}")
        print(f"F1 Score (Micro) = {f1_score_micro:.4f}")
        print(f"F1 Score (Macro) = {f1_score_macro:.4f}")
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', default=Path('/home/shivani/work/bert'), type=Path, 
        help='give the path of the work directory')
    parser.add_argument('--data-dir', default=Path('/home/shivani/work/data/new_shuf_data.tsv'), type=Path, 
        help='give the path of the data directory')
    parser.add_argument('--cfg', default=Path('/home/shivani/work/bert'), 
        type=Path, help='give the config file')
    parser.add_argument('--name', default=str('bert experiment'), 
        type=str, help='give the name to the file')
    args = parser.parse_args()
    return vars(args)

def main(args):
    model_path=args['work_dir']
    trainer = Trainer(data_dir=args['data_dir'])
    df, all_categories = trainer.create_data()
    unique_usda_id = list(all_categories)

    id2label = {k:v for k, v in enumerate(unique_usda_id)}
    label2id = {v:k for k, v in enumerate(unique_usda_id)}

    usda_ids = []
    for i in range(len(df)):
        list1 = [0] * len(id2label)
        labels = df.iloc[i]['usda_id']
        labels = eval(labels)
        for label in labels:
            val = label2id[label]
            list1[int(val)] = 1
        usda_ids.append(list1)

    df = df.assign(usda_id=usda_ids)

    train_dataset, validation_dataset, test_dataset = np.split(df.sample(frac=1, random_state=42), 
                       [int(.8*len(df)), int(.9*len(df))])

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    print("VALIDATION Dataset: {}".format(validation_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
    validation_set = CustomDataset(validation_dataset, tokenizer, MAX_LEN)
    # testing_set = training_set 

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    validation_params = {'batch_size': TEST_BATCH_SIZE,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    validation_loader = DataLoader(validation_set, **validation_params)

    model = BERTClass(id2label)
    model.to(device)


    LEARNING_RATE = 1e-03
    trainable_params = model.head.parameters()
    optimizer = torch.optim.Adam(params =  trainable_params, lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    for epoch in range(EPOCHS):
        train_loop(model, optimizer, training_loader, epoch, testing_loader, scheduler, loss_fn)
    
    chkpt_file =   model_path  / f'chkpt-exp-01600020.pt'        
    torch.save({
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, chkpt_file)

    evaluate(model_path, model, validation_loader, id2label, validation_dataset)

    
if __name__ == '__main__':
    main(parse_args())

