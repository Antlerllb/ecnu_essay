import json
import torch
import os
import sys
from tqdm import tqdm
from transformers import DataProcessor, BertTokenizerFast
from utils import read_json


class LabelDict(object):
    def __init__(self, label_path):
        # self.schema_path = label_path
        self.label_path = label_path
        self.id2labels={}
        self.label2ids={}
        self.load_schema()
    
    def load_schema(self):
        self.label2ids = read_json(self.label_path)
        for idx, label in enumerate(self.label2ids):
            self.id2labels[idx] = label

    def __len__(self):
        return len(self.label2ids)


class CorrectionDataProcessor(DataProcessor):
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.train_path = config.train_path
        self.val_path = config.val_path
        self.test_path = config.test_path

        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)
        self.label_schema = LabelDict(config.schema_path)

    def get_train_data(self):
        return read_json(self.train_path)
    
    def get_dev_data(self):
        return read_json(self.val_path)
    
    def get_test_data(self):
        return read_json(self.test_path)

    def create_dataloader(self, datas, batch_size, shuffle=False, max_length=512):
        tokenizer = self.tokenizer
    
        text = []
        student_text = []
        tgt_text = []
        labels = []
        sent_ids = []
        coarse_labels = []
        for d in tqdm(datas):
            sent_ids.append(int(d['sent_id']))
            text.append(d['sent'])

            if 'fine_grained_error_type' in d.keys() and (d['fine_grained_error_type'] == ["正确"] or d['fine_grained_error_type'] == []):
                tgt_text.append(d['sent'])
            else:

                tgt_text.append(d['revisedSent'])

        max_length = min(max_length, max([len(s) for s in text]))
        print("max sentence length: ", max_length)


        tgt_inputs = tokenizer(
            tgt_text,
            padding=True,
            return_tensors='pt',
            max_length=max_length,
            truncation=True
        )

        inputs = tokenizer(
            text,
            padding=True,
            return_tensors='pt',
            max_length=max_length,
            truncation=True
        )

        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(inputs["input_ids"]),
            torch.LongTensor(inputs["token_type_ids"]),
            torch.LongTensor(inputs["attention_mask"]),
            torch.LongTensor(tgt_inputs["input_ids"]),
            torch.LongTensor(tgt_inputs["attention_mask"]),
            torch.LongTensor(sent_ids)
        )


        dataloader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=4)

        return dataloader

class ClassifyDataProcessor(DataProcessor):
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.train_path = config.train_path
        self.val_path = config.val_path
        self.test_path = config.test_path

        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)
        self.label_schema = LabelDict(config.schema_path)

    def get_train_data(self):
        return read_json(self.train_path)
    
    def get_dev_data(self):
        return read_json(self.val_path)
    
    def get_test_data(self):
        return read_json(self.test_path)
    
    def create_dataloader(self, datas, batch_size, shuffle=False, max_length=512):
        tokenizer = self.tokenizer
        text = []
        labels = []
        sent_ids = []
        tag = 0
        for d in datas:
            if self.config.task == 'score':
                sent_ids.append(int(d['id']))
                text.append('\n'.join(d['text']))
                label = self.label_schema.label2ids[d['essay_score_level']]

            else:   # coarse fine correction
                if type(d['sent_id']) == list:
                    sent_ids.append(sum([int(i) for i in d['sent_id']]))
                else:
                    sent_ids.append(int(d['sent_id']))
 
                text.append(d['sent'])
                label = [0] * len(self.label_schema)
                if self.config.task == 'coarse':
                    key = 'coarse_grained_error_type'
                else:
                    key = 'fine_grained_error_type'

                for l in d[key]:
                    label[self.label_schema.label2ids[l]] = 1

            labels.append(label)
        
        max_length = min(max_length, max([len(s) for s in text]))
        print("max sentence length: ", max_length)

        if self.config.task == 'score':
            inputs = tokenizer(
                text,
                padding=True,
                return_tensors='pt',
                max_length=max_length,
                truncation=True
            )

            dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(inputs["input_ids"]),
                torch.LongTensor(inputs["token_type_ids"]),
                torch.LongTensor(inputs["attention_mask"]),
                torch.LongTensor(labels),
                torch.LongTensor(sent_ids)
            )
        else:
            inputs = tokenizer(
                text,
                padding=True,
                return_tensors='pt',
                max_length=max_length,
                truncation=True
            )
            dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(inputs["input_ids"]),
                torch.LongTensor(inputs["token_type_ids"]),
                torch.LongTensor(inputs["attention_mask"]),
                torch.LongTensor(labels),
                torch.LongTensor(sent_ids)
            )

        dataloader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=4)

        return dataloader