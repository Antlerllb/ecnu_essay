import sys
import os
import torch
import json
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
# import wandb
from transformers import BertTokenizer, GPT2LMHeadModel

import utils
from model import CorrectionPredictor, IdentificationPredictor
from model import IdentificationModel, CorrectionModel


utils.set_random_seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TOKENIZERS_PARALLELISM"] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DATA_DIR = './datas/'
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
    parser.add_argument("--task", type=str, default="coarse", choices=["coarse", "fine", "score","correction"], help="optimizer")
    parser.add_argument("--use_ref_input", type=utils.str2bool, default=True, help="optimizer")

    parser.add_argument("--version", type=str, default="fineII_0320", help="whether to use bert training or not (default: True)")
    parser.add_argument("--error_only", type=utils.str2bool, default=True, help="train the NER model or not (default: False)")
    parser.add_argument("--is_course_grained", type=utils.str2bool, default=False, help="train the NER model or not (default: False)")

    # bart
    parser.add_argument("--accumulate_grads", type=int, default=1, help="accumulate_grads")
    parser.add_argument("--num_beams", type=int, default=4, help="whether to use bert training or not (default: True)")

    # 下面参数基本默认
    parser.add_argument("--train_path", type=str, default=os.path.join(DATA_DIR, 'train_r.json'),
                        help="train_path")
    parser.add_argument("--dev_path", type=str, default=os.path.join(DATA_DIR, 'dev_r.json'),
                        help="dev_path")
    parser.add_argument("--test_path", type=str, default=os.path.join(DATA_DIR, 'test_e.json'),
                        help="dev_path")
    parser.add_argument("--train_num", type=int, default=-1,help="train data number")
    parser.add_argument("--dev_num", type=int, default=-1,help="dev data number")
    parser.add_argument("--schema_path", type=str, default=os.path.join(DATA_DIR, 'label2id.json'),
                        help="schema_path")
    parser.add_argument("--coarse_schema_path", type=str, default=os.path.join(DATA_DIR, 'course_label2id.json'),
                        help="schema_path")

    parser.add_argument("--results_path", type=str, default=os.path.join(BASE_DIR, 'results'),
                        help="result_path")
    # parser.add_argument("--ckpt_save_path", type=str,
    #                     default=os.path.join(BASE_DIR, 'weights/0313'), help="ckpt_save_path")
    parser.add_argument("--resume_ckpt", type=str,
                        default=None, help="checkpoint file name for resume")
    parser.add_argument("--pretrained_path", type=str,
                        default="/home/xinshu/pt/bert-base-chinese", help="pretrained_path")

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
        ckpt_save_path = os.path.join(BASE_DIR, './lightning_logs', args.version)
        ckpt_callback = ModelCheckpoint(
            dirpath=ckpt_save_path,                           # 模型保存路径
            filename=args.ckpt_name + "_{epoch}_{val_f1:.4f}",     # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
            monitor='val_f1',                                      # 根据验证集上的准确率评估模型优劣
            mode='max',
            save_top_k=3,                                          # 保存得分最高的前两个模型
            verbose=True,
        )

        resume_checkpoint=None
        if args.resume_ckpt:
            resume_checkpoint=os.path.join(ckpt_save_path, args.resume_ckpt)   # 加载已保存的模型继续训练
        
        # logger = TensorBoardLogger(
        #     project='coling',
        #     save_dir="./lightning_logs/",
        #     # log_model=args.version,
        #     version=args.version,
        #     # offline=True,
        #     )

        logger = TensorBoardLogger(
        save_dir="./lightning_logs/",
        name=None, # 指定experiment, ./lightning_logs/exp_name/version_name
        # version=args.params_dict['environment']['serialization_dir']+'_node',  # 指定version, ./lightning_logs/version_name
        version=args.version,
        )
        
        # 设置训练器
        if not os.path.exists(os.path.join('./lightning_logs', args.version)):
            os.makedirs(os.path.join('./lightning_logs', args.version))
        
        # es = EarlyStopping('train_loss', patience=10, mode='min')
        trainer = pl.Trainer(
            progress_bar_refresh_rate=1,
            logger=logger,
            resume_from_checkpoint = resume_checkpoint,
            max_epochs=args.max_epochs,
            callbacks=[ckpt_callback],
            limit_train_batches=2000,
            limit_val_batches=1000,
            checkpoint_callback=True,
            gpus=1
        )

        # 开始训练模型
        trainer.fit(model)
    else:   
        # ============= test 测试模型==============
        print("\nstart test model...")

        dir_name = os.path.join(args.results_path, args.version)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        outfile_txt = os.path.join(dir_name, args.test_ckpt_name[:-5] + ".json")

        # 开始测试，将结果保存至输出文件
        # checkpoint_path模型路径
        ckpt_save_path = os.path.join(BASE_DIR, './lightning_logs', args.version)
        checkpoint_path = os.path.join(ckpt_save_path, args.test_ckpt_name)
        # print(checkpoint_path)

        if args.task == 'correction':
            predictor = CorrectionPredictor(checkpoint_path, args)
        else:
            predictor = IdentificationPredictor(checkpoint_path, args)

        predictor.generate_result(outfile_txt)

        # sentences = ['我家阳台上有一盆绿萝，绿萝现在长了几根藤，这几根藤现在短短的，绿绿的。', '我心里非常兴奋，很期待绿萝长大后是什么样子的。', '过了几天绿萝的"头发"长长了。', '我想起它的"头发"还是短短的时候一对比，真的是好看了很多啊！', '那绿萝现在又是什么样子呢？', '可好看了，那几根藤好像姑娘的染了个绿色的长头发。', '叶子它们一个个小巧玲珑，只有一个乒乓球那么大。', '那些叶子一个个油光华亮的，好像涂了一层油似的。', '摸上去又凉又滑，舒服的很。', '绿萝也是很好养的有一次我忘记给它浇水了，但它还是长的很好，可以说绿萝是很有生命力的。', '绿萝它不但叶子长的快，它的茎叶长的很快，几个月过后绿萝的茎部长出了花盆我只好给它换盆子。', '我还发现了绿萝的茎上有好几个黄色的"戒指"可真是一个隐藏的"大富豪"啊！', '几年过后这盆绿萝的腾也是长长了很多格外好着以前像头发的腾现在更像柳树的枝叶了。', '绿萝的生命力真的是太惊人了，绿萝这顽强的品德值得我们学习！', '']
        # sentences = [sent for sent in sentences if sent != '']

        # predictor.predict(sentences)

        # predictor.generate_result(outfile_txt)
        print('\n', 'outfile_txt name:', outfile_txt)

        output_list = []
