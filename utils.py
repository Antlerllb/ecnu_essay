from pathlib import Path
import pandas as pd
import json
import random
import re
from typing import List, OrderedDict
import argparse
import torch
import numpy as np


def get_path(*path: str):
    project_dir = Path(__file__).resolve().parent
    return project_dir.joinpath(*path)


def get_data(*path: str):
    if not path:
        return get_path('datas')
    return get_path('datas').joinpath(*path)


def get_result(*path: str):
    if not path:
        return get_path('results')
    return get_path('results').joinpath(*path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    """将字符转化为bool类型"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_tsv(path):
    """读取csv、tsv文件，并转换成dict格式"""
    return pd.read_csv(path, sep='\t', header=0, index_col=0).to_dict("records")

def save_json(R, path, **kwargs):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(R, f, indent=2, ensure_ascii=False, **kwargs)
    print(f"{path} saved with {len(R)} samples!")


def save_txt(path, data):
    with open(path, 'w') as f:
        f.write('\n'.join(data))
    print(f"{path} saved with {len(data)} samples!")

def read_txt(path):
    with open(path, 'r') as f:
        return [l.strip() for l in f.readlines()]

def read_json(path):
    with open(path, 'r') as fjson:
        return json.load(fjson, object_pairs_hook=OrderedDict)


def convert_mydatas2para(filename, datas):
    R = [f"{d['sent_id']}\t{d['sent']}\t{d['revisedSent']}" for d in datas] # if d['errorType'] != ['正确']]
    save_txt(filename, R)

