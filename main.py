import sys
import os
from os.path import exists

import torch
import json
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
# import wandb
from transformers import BertTokenizer, GPT2LMHeadModel

import utils
from model import IdentificationModel, CorrectionModel
from utils import get_data, get_result, get_path

utils.set_random_seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# wandb.init(project='course_sicksent')


def gen_args():
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=utils.str2bool, default=True, help="train the NER model or not (default: False)")
    parser.add_argument("--batch_size", type=int, default=16, help="input batch size for training and test (default: 8)")
    parser.add_argument("--max_epochs", type=int, default=100, help="the max epochs for training and test (default: 5)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (default: 2e-5)")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout (default: 0.2)")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="optimizer")
    parser.add_argument("--task", type=str, default="fine", choices=["coarse", "fine", "score","correction"], help="optimizer")
    parser.add_argument("--use_ref_input", type=utils.str2bool, default=True, help="optimizer")

    parser.add_argument("--version", type=str, default="fineII_0320", help="whether to use bert training or not (default: True)")
    parser.add_argument("--error_only", type=utils.str2bool, default=True, help="train the NER model or not (default: False)")
    parser.add_argument("--is_course_grained", type=utils.str2bool, default=False, help="train the NER model or not (default: False)")

    # bart
    parser.add_argument("--accumulate_grads", type=int, default=1, help="accumulate_grads")
    parser.add_argument("--num_beams", type=int, default=4, help="whether to use bert training or not (default: True)")

    # 下面参数基本默认
    parser.add_argument("--train_path", type=str, default=get_data('exp/train.json'),
                        help="train_path")
    parser.add_argument("--val_path", type=str, default=get_data('exp/val.json'),
                        help="val_path")
    parser.add_argument("--test_path", type=str, default=get_data('exp/test.json'),
                        help="test_path")
    parser.add_argument("--train_num", type=int, default=-1,help="train data number")
    parser.add_argument("--dev_num", type=int, default=-1,help="dev data number")
    parser.add_argument("--schema_path", type=str, default=get_data('datas/archives/label2id.json'),
                        help="schema_path")
    parser.add_argument("--coarse_schema_path", type=str, default=get_data('datas/archives/course_label2id.json'),
                        help="schema_path")

    parser.add_argument("--results_path", type=str, default=get_result(),
                        help="result_path")
    # parser.add_argument("--ckpt_save_path", type=str,
    #                     default=os.path.join(BASE_DIR, 'weights/0313'), help="ckpt_save_path")
    parser.add_argument("--resume_ckpt", type=str,
                        default=None, help="checkpoint file name for resume")
    parser.add_argument("--pretrained_path", type=str,
                        default="/sshfs/pretrains/bert-base-chinese", help="pretrained_path")

    parser.add_argument("--ckpt_name",  type=str, default="small_lion", help="ckpt save name")
    parser.add_argument("--test_ckpt_name",  type=str, default="small_lion_epoch=44_val_f1=77.9.ckpt", help="ckpt name for test")
    parser.add_argument("--external_predict", type=utils.str2bool, default=False, help="external predict")

    # 与correction评测相关的参数
    parser.add_argument("-d", "--device", type=int, help="The ID of GPU", default=0)
    parser.add_argument("-w", "--worker_num", type=int, help="The number of workers", default=16)
    parser.add_argument("-m", "--merge", help="Whether merge continuous replacement/deletion/insertion", action="store_true")
    parser.add_argument("-s", "--multi_cheapest_strategy", type=str, choices=["first", "all"], default="all")
    parser.add_argument("--segmented", help="Whether tokens have been segmented", action="store_true")  # 支持提前token化，用空格隔开
    parser.add_argument("--no_simplified", help="Whether simplifying chinese", action="store_true")  # 将所有corrections转换为简体中文
    parser.add_argument("--beta",help="Value of beta in F-score. (default: 0.5)",default=0.5,type=float)
    parser.add_argument("-v","--verbose",help="Print verbose output.",action="store_true")
    eval_type = parser.add_mutually_exclusive_group()
    eval_type.add_argument(
        "-dt",
        help="Evaluate Detection in terms of Tokens.",
        action="store_true")
    eval_type.add_argument(
        "-ds",
        help="Evaluate Detection in terms of Spans.",
        action="store_true")
    eval_type.add_argument(
        "-cs",
        help="Evaluate Correction in terms of Spans. (default)",
        action="store_true")
    eval_type.add_argument(
        "-cse",
        help="Evaluate Correction in terms of Spans and Error types.",
        action="store_true")
    parser.add_argument(
        "-single",
        help="Only evaluate single token edits; i.e. 0:1, 1:0 or 1:1",
        action="store_true")
    parser.add_argument(
        "-multi",
        help="Only evaluate multi token edits; i.e. 2+:n or n:2+",
        action="store_true")
    parser.add_argument(
        "-multi_hyp_avg",
        help="When get multiple hypotheses for a sentence, calculate their average F-scores for this sentence.",
        action="store_true")  # For IAA calculation
    parser.add_argument(
        "-multi_hyp_max",
        help="When get multiple hypotheses for a sentence, calculate their F-scores and select the max one for this sentence.",
        action="store_true")    # For multiple hypotheses system evaluation
    parser.add_argument(
        "-filt",
        help="Do not evaluate the specified error types.",
        nargs="+",
        default=[])
    parser.add_argument(
        "-cat",
        help="Show error category scores.\n"
            "1: Only show operation tier scores; e.g. R.\n"
            "2: Only show main tier scores; e.g. NOUN.\n"
            "3: Show all category scores; e.g. R:NOUN.",
        choices=[1, 2, 3],
        type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = gen_args()

    print('--------config----------')
    print(args)
    print('--------config----------')

    if args.is_train == True:
        #-----------训练模型-----------
        print("start train model ...")
        if args.task == 'correction':
            model = CorrectionModel(args)
        else:
            model = IdentificationModel(args)

        # 设置保存模型的路径及参数
        ckpt_save_path = get_path('lightning_logs', args.version)
        if not exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        ckpt_callback = ModelCheckpoint(
            dirpath=ckpt_save_path,  # 模型保存路径
            filename=args.ckpt_name + "_{epoch}_{val_f1:.4f}",  # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
            monitor='val_f1',  # 根据验证集上的准确率评估模型优劣
            mode='max',
            save_top_k=3,  # 保存得分最高的前两个模型
            verbose=True,
        )

        resume_checkpoint = None
        if args.resume_ckpt:
            resume_checkpoint = os.path.join(ckpt_save_path, args.resume_ckpt)  # 加载已保存的模型继续训练

        # logger = TensorBoardLogger(
        #     project='coling',
        #     save_dir="./lightning_logs/",
        #     # log_model=args.version,
        #     version=args.version,
        #     # offline=True,
        #     )

        logger = TensorBoardLogger(
            save_dir=get_path('lightning_logs'),
            name=None,  # 指定experiment, ./lightning_logs/exp_name/version_name
            # version=args.params_dict['environment']['serialization_dir']+'_node',  # 指定version, ./lightning_logs/version_name
            version=args.version,
        )

        # es = EarlyStopping('train_loss', patience=10, mode='min')
        trainer = pl.Trainer(
            # progress_bar_refresh_rate=1,
            logger=logger,
            # resume_from_checkpoint=resume_checkpoint,
            max_epochs=args.max_epochs,
            callbacks=[ckpt_callback],
            limit_train_batches=2000,
            limit_val_batches=1000,
            # checkpoint_callback=True,
            # gpus=1
            devices=1,  # 只使用一个设备
            num_nodes=1,  # 只使用一个节点
        )

        # 开始训练模型
        trainer.fit(model)
