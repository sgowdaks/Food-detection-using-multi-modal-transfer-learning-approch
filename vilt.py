import logging as log
import pandas as pd
import numpy as np
import pprint
import torch
import csv
import argparse

from PIL import Image
from transformers import ViltProcessor
from transformers import ViltConfig
from transformers import ViltForQuestionAnswering
from torch.utils.data import DataLoader
from tqdm import tqdm 
from pathlib import Path


log.basicConfig(level=log.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_data(data_dir):

    main_usda_id = {}

    column_names =  ['diaries', 'usda_id', 'image_id']
    df_usda = pd.DataFrame(columns = column_names)

    with open(data_dir, 'r') as input:
        csv_reader =  csv.reader(input, delimiter='\t')
        for line in csv_reader:
            dicti = pd.DataFrame({'diaries': line[0], 'usda_id': line[1], 'image_id': line[2]}, index=[0])
            for i in eval(line[1]):
                if i in main_usda_id:
                    main_usda_id[i] += 1
                else:
                    main_usda_id[i] = 1
            df_usda = pd.concat([df_usda, dicti], ignore_index = True, axis = 0) 
    return df_usda, main_usda_id

def id_to_filepath(img_dir, id_):
    return img_dir / f'image{id_}.png'


class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, diaries, usda_id, image_id, processor, id2imagename, config):
        self.diaries = diaries
        self.usda_id = usda_id
        self.image_id = image_id
        self.processor = processor
        self.id2imagename = id2imagename
        self.config = config

    def __len__(self):
        return len(self.usda_id)

    def __getitem__(self, idx):
        usda_id = self.usda_id.iloc[idx]
        diaries = self.diaries.iloc[idx]
        image = self.image_id.iloc[idx]
        image = self.id2imagename[int(image)]
        image = Image.open(image).convert('RGB')
        
        
        encoding = self.processor(image, diaries, padding="max_length", truncation=True, return_tensors="pt")

        for k,v in encoding.items():
            encoding[k] = v.squeeze()
            
        targets = torch.zeros(len(self.config.id2label))
        usda_id = eval(usda_id)
        if len(usda_id) > 1:
            for id in usda_id:
                label =  self.config.label2id[id]
                targets[int(label)] = 1.0      
        else:
            if len(usda_id) == 0:
                print(diaries)
            label = self.config.label2id[usda_id[0]]
            targets[int(label)] = 1.0
        
        encoding["labels"] = targets

        return encoding
    


def training_evaluation(model, optimizer, train_dataloader, val_dataloader, num_epochs, model_save_path):

    train_loss = []
    val_loss = []

    for i in range(num_epochs):
        print("epoch : ", i)
        loss_for_each_epoch = []
        model.train()
        for batch in tqdm(train_dataloader):
            batch = {k:v.to(device) for k,v in batch.items()}
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss_for_each_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
        average_ = sum(loss_for_each_epoch)/len(loss_for_each_epoch)
        train_loss.append(average_)
        print(f"Training loss: {average_:.4f}")
            
        model.eval()
        val_loss_for_each_epoch = []
        for batch in tqdm(val_dataloader):
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.no_grad():        
                outputs = model(**batch)
            loss = outputs.loss
            val_loss_for_each_epoch.append(loss.item())
        average = sum(val_loss_for_each_epoch)/len(val_loss_for_each_epoch)
        val_loss.append(average)
            
        print(f"validataion loss: {average:.4f}")

        chkpt_file = model_save_path / f'chkpt-exp-num{num_epochs:05d}.pt'        
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, chkpt_file)

    
def getlabels(data):
    usda_id = data["usda_id"]
    txt_ = eval(str(usda_id))
    each_usda_id = []
    for t in txt_:
        each_usda_id.append(t) 
    return each_usda_id

def add_related_categories(predictions, probs, config):
    predictions[np.where(probs > 0.01)] = 1
    predicted_labels = [config.id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    return predicted_labels

        
def evaluate_model(df_test, dataset_test, model, config, load_check_point_from, output_results):
    sigmoid = torch.nn.Sigmoid()
    state_dicti = torch.load(Path(load_check_point_from))
    model.load_state_dict(state_dicti['model_state_dict'])
    model = model.to(device)
    model.eval()
    with open(output_results, 'wt') as out_file:  
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['ground truth', 'prediction'])
        for num in range(len(df_test)):
            weights = []
            example = dataset_test[num]
            ground_truth = getlabels(df_test.iloc[num])
            example = {k: v.unsqueeze(0).to(device) for k,v in example.items()}
            outputs = model(**example)
            loss = outputs.loss
            logits = outputs.logits
            probs = sigmoid(logits.squeeze()).to("cpu")
            for i in probs:
                weights.append(i)

            predictions = np.zeros(probs.shape)  
            new_categories = add_related_categories(predictions = np.zeros(probs.shape), probs = probs, config = config)  
            predictions[np.where(probs > 0.1)] = 1
            predicted_labels = []
            for idx, label in enumerate(predictions):
                if label == 1.0:
                    config.id2label[idx]
            if len(predicted_labels) == 0:
                probs = probs.tolist()
                max_val = max(probs)
                for i, val in enumerate(probs):
                    if val == max_val:
                        predicted_labels.append(config.id2label[i])



            print("predicted_labels: ", predicted_labels)
            print("answer: ", ground_truth)
            
            tsv_writer.writerow([ground_truth, predicted_labels, new_categories])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', default=Path('/home/shivani/work/bert'), type=Path, 
        help='give the path of the work directory')
    parser.add_argument('--data-dir', default=Path('/home/shivani/work/data/new_proper_branded.tsv'), type=Path, 
        help='give the path of the data directory')
    parser.add_argument('--img-dir', default=Path('/home/cgusti/CLIP/images_166/'), type=Path, 
        help='give the path of the image data directory')
    parser.add_argument('--test-dir', default=Path('/home/shivani/work/data/'), type=Path, 
        help='give the path of the test data directory')
    parser.add_argument('--model-save-path', default=Path('/home/shivani/work/data/'), type=Path, 
        help='give the path of where the checkpoints should be saved')
    parser.add_argument('--cfg', default=Path('/home/shivani/work/vilt'), 
        type=Path, help='give the config file')
    parser.add_argument('--load-checkpoint', default=Path('/home/shivani/work/vilt'), 
        type=Path, help='give the path to checkpoint file, give the last epochs name like chkpt-exp-num<num of epochs - 1>.pt')
    parser.add_argument('--ouput-results', default=Path('/home/shivani/work/vilt'), 
        type=Path, help='give the path where you want to save the output results to')
    parser.add_argument('--name', default=str('bert experiment'), 
        type=str, help='give the name to the file')
    args = parser.parse_args()
    return vars(args)

def main(args):
    log.info("creating datasets")

    df_usda, usda_id = create_data(data_dir=args['data_dir'])

    category = []
    for key in usda_id.keys():
        if key not in category:
            category.append(key)

    id2label = {}
    label2id = {}
    for i, val in enumerate(category):
        id2label[i] = val
        label2id[val] = i

    id2imagename = {}

    for i in range(1, 18000):
        id2imagename[i] = id_to_filepath(img_dir=args['img_dir'], id_=i)

    mydict = {}
    for i in range(len(df_usda)):
        k = df_usda.iloc[i]["usda_id"]
        k = eval(k)
        for j in k:
            if j in mydict:
                mydict[j] += 1
            else:
                mydict[j] = 1
    
    log.info(f"found {len(mydict)} recs; {len(mydict)} classes")

    # !git lfs install
    # !git clone https://huggingface.co/dandelin/vilt-b32-finetuned-vqa

    config = ViltConfig.from_pretrained(args['cfg'] / "vilt-b32-finetuned-vqa_1")


    df_train, df_val, df_test = np.split(df_usda.sample(frac=1, random_state=42), 
                       [int(.8*len(df_usda)), int(.9*len(df_usda))])
    
    with open(args['test_dir'], "w") as input:
        csv_writer = csv.writer(input, delimiter="\t")
        csv_writer.writerow(["dairies", "usda_id", "image_id"])
        for i in range(len(df_test)):
            dairies, usda_id, image_id = df_test.iloc[i]
            csv_writer.writerow([dairies, usda_id, image_id])


    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    dataset_train = VQADataset(diaries=df_train["diaries"][:100],
                        usda_id=df_train["usda_id"][:100],
                        image_id=df_train["image_id"][:100],
                        processor=processor,
                        id2imagename = id2imagename,
                        config = config
                        )

    dataset_val = VQADataset(diaries=df_val["diaries"][:100],
                        usda_id=df_val["usda_id"][:100],
                        image_id=df_val["image_id"][:100],
                        processor=processor,
                        id2imagename = id2imagename,
                        config = config
                        )

    dataset_test = VQADataset(diaries=df_test["diaries"],
                        usda_id=df_test["usda_id"],
                        image_id=df_test["image_id"],
                        processor=processor,
                        id2imagename = id2imagename,
                        config = config
                        )

    model = ViltForQuestionAnswering(config)

    num_classes = len(config.id2label)   
    model.classifier._modules['3'] = torch.nn.Linear(1536, num_classes)
    model = model.to(device)

    def optimizer_prameters(param_optimizer, freeze_encoder = True, un_freeze=3):
        requires_grad = []
        no_decay = ['bias', 'gamma', 'beta']
        max_ = 0
        for n, p in param_optimizer:
            if n.startswith("vilt.encoder.layer."):
                max_ = max(max_, int(n.split(".")[3]))           

        if freeze_encoder:
            if un_freeze == 0:
                param_optimizer = [(n, p) for n, p in param_optimizer if not n.startswith("vilt.")]
            else:
                for i in range(max_, max_ - un_freeze, -1):
                    name = "vilt.encoder.layer." + str(i)
                    list1 = []
                    for n, p in param_optimizer:
                        if not n.startswith(name):
                            list1.append((n, p))
                        else:
                            requires_grad.append(n)
                param_optimizer = list1

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
            'weight_decay_rate': 0.01
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay_rate': 0.0}]

        return optimizer_grouped_parameters

    # optimizer = AdamW(optimizer_prameters(), lr = float(self.cfg['learning_rate']))  
    param_optimizer = list(model.named_parameters())
    optimizer = torch.optim.AdamW(optimizer_prameters(param_optimizer), lr=1e-04)

    
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        pixel_values = [item['pixel_values'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # create padded pixel values and corresponding pixel mask
        encoding = processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        
        # create new batch
        batch = {}
        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_mask)
        batch['token_type_ids'] = torch.stack(token_type_ids)
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = torch.stack(labels)
        return batch

    train_dataloader = DataLoader(dataset_train, collate_fn=collate_fn, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(dataset_val, collate_fn=collate_fn, batch_size=32, shuffle=True )
    training_evaluation(model, optimizer, train_dataloader, val_dataloader,  num_epochs = 1, model_save_path=args['model_save_path'])
    evaluate_model(df_test, dataset_test, model, config, args['load_checkpoint'], args['output_results'])

if __name__ == '__main__':
    main(parse_args())
