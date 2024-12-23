import os

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support, cohen_kappa_score, accuracy_score
from transformers import BertForSequenceClassification, BertTokenizer, BertModel, BartForConditionalGeneration, AutoConfig
from transformers.models.bart.modeling_bart import BartClassificationHead
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl

from utils import *
from evaluation import MultiLabel_evaluate, bleu, levenshtein, bert_score, bert_ppl, cherrant, rewrite_eval, em_eval, New_multilabel_evaluate
from data_process import ClassifiyDataProcessor, CorrectionDataProcessor


class IdentificationModel(pl.LightningModule):
    def __init__(self, config):
        super(IdentificationModel, self).__init__()

        self.config = config
        self.save_hyperparameters()

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.dropout = config.dropout
        self.optimizer = config.optimizer

        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)
        self.processor = ClassifiyDataProcessor(config, self.tokenizer)

        self.labels = len(self.processor.label_schema.id2labels)
        self.model = BertModel.from_pretrained(config.pretrained_path)
        self.classifier_dropout = (
            config.classifier_dropout if self.model.config.classifier_dropout is not None else self.model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, self.labels * 2)

        self.loss_fct = nn.CrossEntropyLoss()

        self.model.resize_token_embeddings(len(self.tokenizer))
        print('num tokens:', len(self.tokenizer))

        self.gold_corpus = []

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.dev_loader

    def prepare_data(self):
        train_data = self.processor.get_train_data()
        dev_data = self.processor.get_dev_data()

        if self.config.train_num > 0:
            train_data = train_data[:self.config.train_num]
        if self.config.dev_num > 0:
            dev_data = dev_data[:self.config.dev_num]

        print("train_length:", len(train_data))
        print("valid_length:", len(dev_data))

        self.train_loader = self.processor.create_dataloader(train_data, batch_size=self.batch_size, shuffle=True)
        self.dev_loader = self.processor.create_dataloader(dev_data, batch_size=self.batch_size, shuffle=False)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        features = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = features[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits.reshape(logits.shape[0], -1, 2)

    def training_step(self, batch, batch_idx):
        inputs_ids, token_type_ids, attention_mask, labels, sent_ids = batch
        logits = self(inputs_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

        # logits = self.classifier(pooled_output)

        loss = self.loss_fct(logits.reshape(-1, 2), labels.view(-1))

        self.log("train_classify_loss", loss.item())
        self.log('train_loss', loss.item())
        # wandb.log({'train_loss':feats.loss.item()})
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inputs_ids, token_type_ids, attention_mask, labels, sent_ids = batch
        logits = self(inputs_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

        # logits = self.classifier(pooled_output)

        loss = self.loss_fct(logits.reshape(-1, 2), labels.view(-1))

        gold = labels
        pre = (F.softmax(logits, dim=-1)[:, :, -1] > 0.5).int()

        self.gold_corpus.append(inputs_ids)

        return loss.cpu(), gold.cpu(), pre.cpu()

    def validation_epoch_end(self, outputs):
        val_loss, gold, pre = zip(*outputs)

        val_loss = torch.stack(val_loss).mean()
        gold = torch.cat(gold)
        pre = torch.cat(pre)

        macrof1, microf1 = New_multilabel_evaluate(pre, gold, verbose=True,
                                                   id2labels=self.processor.label_schema.id2labels,
                                                   error_only=self.config.error_only)

        true_seqs = [[self.processor.label_schema.id2labels[int(idx)] for idx in torch.nonzero(g).squeeze(1)] for g in
                     gold]
        pred_seqs = [[self.processor.label_schema.id2labels[int(idx)] for idx in torch.nonzero(p).squeeze(1)] for p in
                     pre]

        print(f"pred seq len: {len(pred_seqs)}, gold seq len: {len(true_seqs)}")

        self.log('val_f1', microf1 + macrof1)

        self._save_dev_result(true_seqs, pred_seqs)

    def _save_dev_result(self, true_seqs, pred_seqs):
        logdir = os.path.join(self.trainer.logger.save_dir, self.trainer.logger.version)
        gt = []
        for g in self.gold_corpus:
            gt.extend(self.tokenizer.batch_decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        R = []
        for g, t, p in zip(gt, true_seqs, pred_seqs):
            R.append(dict(
                **{
                    'text': g,
                    'expected': t,
                    'generated': p}
            ))
        filename = os.path.join(logdir, f"dev_epoch{self.current_epoch:02}.json")
        save_json(R, filename)
        return

    def configure_optimizers(self):
        arg_list = [p for p in self.parameters() if p.requires_grad]
        print("Num parameters:", len(arg_list))
        if self.optimizer == 'Adam':
            return torch.optim.Adam(arg_list, lr=self.lr, eps=1e-8)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(arg_list, lr=self.lr, momentum=0.9)


class CorrectionModel(pl.LightningModule):
    def __init__(self, config):
        super(CorrectionModel, self).__init__()

        self.config = config
        self.save_hyperparameters()

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.dropout = config.dropout
        self.optimizer = config.optimizer

        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)
        self.processor = CorrectionDataProcessor(config, self.tokenizer)

        self.num_labels = len(self.processor.label_schema.id2labels)

        self.model = BartForConditionalGeneration.from_pretrained(config.pretrained_path)

        # self.model.config.problem_type = 'multi_label_classification'
        self.model.resize_token_embeddings(len(self.tokenizer))
        print('num tokens:', len(self.tokenizer))

        self.gold_corpus = []
        self.val_sent_ids = []

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.dev_loader

    def prepare_data(self):
        train_data = self.processor.get_train_data()
        dev_data = self.processor.get_dev_data()

        if self.config.train_num > 0:
            train_data = train_data[:self.config.train_num]
        if self.config.dev_num > 0:
            dev_data = dev_data[:self.config.dev_num]

        print("train_length:", len(train_data))
        print("valid_length:", len(dev_data))

        self.train_loader = self.processor.create_dataloader(train_data, batch_size=self.batch_size, shuffle=True)
        self.dev_loader = self.processor.create_dataloader(dev_data, batch_size=self.batch_size, shuffle=False)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tgt_inputs_ids=None, tgt_attention_mask=None,
                labels=None):
        if tgt_inputs_ids is not None:
            target_ids = tgt_inputs_ids[:, :-1].contiguous()
            lm_labels = tgt_inputs_ids[:, 1:].clone()
            lm_labels[tgt_inputs_ids[:, 1:] == self.model.config.pad_token_id] = -100
        else:
            target_ids, lm_labels = None, None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_ids,
            labels=lm_labels,
            output_hidden_states=True
        )
        return outputs

    def training_step(self, batch, batch_idx):
        inputs_ids, token_type_ids, attention_mask, tgt_inputs_ids, tgt_attention_mask, sent_ids = batch
        outputs = self(inputs_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                       tgt_inputs_ids=tgt_inputs_ids, tgt_attention_mask=tgt_attention_mask)
        self.log("train_generate_loss", outputs.loss.item())

        self.log('train_loss', outputs.loss.item())
        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_idx):
        inputs_ids, token_type_ids, attention_mask, tgt_inputs_ids, tgt_attention_mask, sent_ids = batch
        outputs = self.model.generate(input_ids=inputs_ids, attention_mask=attention_mask,
                                      num_beams=self.config.num_beams, max_length=350, use_cache=True,
                                      return_dict_in_generate=True, output_hidden_states=True, output_scores=True)

        gold, pre = torch.tensor([0]), torch.tensor([0])

        self.gold_corpus.append(inputs_ids)

        pred_ids = outputs[0]

        self.val_sent_ids.append(sent_ids)

        return gold.cpu(), pre.cpu(), pred_ids.cpu(), tgt_inputs_ids.cpu()

    def validation_epoch_end(self, outputs):

        gold, pre, pred_ids, tgt_inputs_ids = zip(*outputs)

        true_seqs, pred_seqs = [], []

        pred_corpus = []
        target_corpus = []
        input_corpus = []
        sent_ids = []
        for p_ids in pred_ids:
            pred_corpus += self.tokenizer.batch_decode(p_ids, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)
        for t_ids in tgt_inputs_ids:
            target_corpus += self.tokenizer.batch_decode(t_ids, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
        for i_ids in self.gold_corpus:
            input_corpus += self.tokenizer.batch_decode(i_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)
        for sent in self.val_sent_ids:
            sent_ids += sent.cpu().tolist()

        print("pred[0]", pred_corpus[0])

        start = len('修改后的句子：')
        # pred_corpus = [p.strip().replace(' ','').split('|')[-1][start:].split('\n')[0] for p in pred_corpus]
        pred_corpus = [p.replace(' ', '').replace('[SEP]', '') for p in pred_corpus]
        target_corpus = [t.replace(' ', '') for t in target_corpus]
        input_corpus = [i.replace(' ', '') for i in input_corpus]
        # input_corpus = ['：'.join(i.split('|')[1].split('：')[1:]).replace(' ','') for i in input_corpus]

        print("pred[0]", pred_corpus[0])
        print("target[0]", target_corpus[0])
        print("input[0]", input_corpus[0])

        char_f1, word_f1 = rewrite_eval(self.config, target_corpus, pred_corpus, sent_ids,
                                        save_path=os.path.join('./lightning_logs', self.config.version,
                                                               f"dev_epoch{self.current_epoch:02}.hyp.para"),
                                        data_path=self.config.dev_path)

        bleu_score = bleu(pred_corpus, target_corpus) * 100
        levenshtein_tgt = levenshtein(pred_corpus, target_corpus)
        levenshtein_input = levenshtein(pred_corpus, input_corpus)
        # bert_s = bert_score(preds=pred_corpus, golds=target_corpus)
        ppl = bert_ppl(pred_corpus)
        em = em_eval(target_corpus, pred_corpus)

        self.log('val_f1', char_f1)
        print(f"the em score: {em * 100:.2f}")
        print(f"the bleu score: {bleu_score:.4f}")
        # print(f"the bert score: {bert_s:.4f}")
        print(f"the bert ppl score: {ppl:.4f}")
        print(f"the levenshtein score with gold seq: {levenshtein_tgt:.4f}")
        print(f"the levenshtein score with input seq: {levenshtein_input:.4f}")

        self.log('val_char_f1', char_f1)
        self.log('val_em', round(em * 100, 2))
        self.log('val_bleu', bleu_score)
        # self.log('val_bert_score',bert_s)
        self.log('val_bert_ppl', ppl)
        self.log('val_levenshtein_tgt', levenshtein_tgt)
        self.log('val_levenshtein_input', levenshtein_input)

        # self._save_dev_result(true_seqs, pred_seqs, pred_corpus, target_corpus)

        self.val_sent_ids, self.gold_corpus = [], []

    def _save_dev_result(self, true_seqs, pred_seqs, pred_corpus, target_corpus):
        logdir = os.path.join('./lightning_logs', self.config.version)
        gt = []
        for g in self.gold_corpus:
            gt.extend(self.tokenizer.batch_decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        R = []
        if true_seqs != [] and pred_seqs != []:
            for g, t, p, pc, tc in zip(gt, true_seqs, pred_seqs, pred_corpus, target_corpus):
                R.append(dict(
                    **{
                        'text': g,
                        'gold label': t,
                        'pred label': p,
                        'expected': tc,
                        'generated': pc}
                ))
        else:
            for g, pc, tc in zip(gt, pred_corpus, target_corpus):
                R.append(dict(
                    **{
                        'text': g,
                        'expected': tc,
                        'generated': pc}
                ))
        filename = os.path.join(logdir, f"dev_epoch{self.current_epoch:02}.json")
        save_json(R, filename)
        return

    def _get_grouped_params(self):
        no_decay = ["bias", "LayerNorm.weight"]

        # Group parameters to those that will and will not have weight decay applied
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def configure_optimizers(self):
        optimizer = optim.AdamW(self._get_grouped_params(), lr=self.config.lr)
        # if self.config.use_warmup:
        #     total_steps = int(len(self.train_dataloader()) // self.config.accumulate_grads ) * self.config.max_epochs # accumulate_grads
        #     warmup_step =  int(total_steps * self.config.warmup_rate)
        #     # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=steps_per_epoch*self.config.max_epochs)
        #     lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)

        #     return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1, 'strict': True, 'monitor': None}]
        # else:
        return optimizer


class IdentificationPredictor:
    def __init__(self, checkpoint_path, config):

        self.model = IdentificationModel.load_from_checkpoint(checkpoint_path, config=config, strict=False)

        self.test_data = self.model.processor.get_test_data()
        self.tokenizer = self.model.tokenizer
        self.config = config
        if not config.external_predict:
            self.dataloader = self.model.processor.create_dataloader(
                self.test_data, batch_size=config.batch_size, shuffle=False)

        print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)

    def predict(self, sentences):

        preds = []
        for sent in sentences:
            inputs = self.model.tokenizer(
                sent,
                padding=True,
                return_tensors='pt',
                max_length=512,
                truncation=True  # 截断
                # return_offsets_mapping=True
            )

            emission = self.model(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
            pred = emission.argmax(dim=-1).cpu()
            result = self.model.processor.label_schema.id2labels[pred.item()]
            preds.append(result)
        return preds

    def generate_result(self, outfile_txt):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        cnt = 0

        with open(outfile_txt, 'w') as fout:
            data = []
            gold = []
            pred_results = []
            # batch:5*8*512, 5是tokenizer有5个结果：input_ids, token_type_ids, attention_mask, offset_mapping和labels
            for batch in tqdm.tqdm(self.dataloader):
                for i in range(len(batch)):
                    batch[i] = batch[i].to(device)
                inputs_ids, token_type_ids, attention_mask, labels, sent_ids = batch

                gold += labels.cpu().tolist()

                emissions = self.model(inputs_ids, token_type_ids, attention_mask)
                preds = (F.softmax(emissions, dim=-1)[:, :, -1] > 0.5).int()

                for pred, inputs, idx in zip(preds, inputs_ids, sent_ids):
                    input_text = self.model.tokenizer.batch_decode(inputs, skip_special_tokens=True,
                                                                   clean_up_tokenization_spaces=True)

                    errorTypes = [self.model.processor.label_schema.id2labels[int(idx)] for idx in
                                  torch.nonzero(pred).squeeze(1)]
                    data.append({
                        'sent_id': int(idx.tolist()),
                        'rawText': ''.join(input_text[1:input_text.index('[ S E P ]')]),
                        'errorType': errorTypes,
                        'revisedSent': ""
                    })
                pred_results += preds.cpu().tolist()

            New_multilabel_evaluate(pred_results, gold, verbose=True,
                                    id2labels=self.model.processor.label_schema.id2labels,
                                    error_only=self.config.error_only)

            json.dump(data, fout, indent=4, ensure_ascii=False)

        print('done--all %d tokens.' % cnt)


class CorrectionPredictor:
    def __init__(self, checkpoint_path, config):

        self.model = CorrectionModel.load_from_checkpoint(checkpoint_path, config=config, strict=False)
        self.test_data = self.model.processor.get_test_data()
        self.config = config
        self.tokenizer = self.model.tokenizer
        if not config.external_predict:
            self.dataloader = self.model.processor.create_dataloader(
                self.test_data, batch_size=config.batch_size, shuffle=False)

        print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)

    def predict(self, sentences):

        preds = []
        for sent in sentences:
            inputs = self.model.tokenizer(
                sent,
                padding=True,
                return_tensors='pt',
                max_length=512,
                truncation=True  # 截断
                # return_offsets_mapping=True
            )

            emission = self.model(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
            pred = emission.argmax(dim=-1).cpu()
            result = self.model.processor.label_schema.id2labels[pred.item()]
            preds.append(result)
        return preds

    def generate_result(self, outfile_txt):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        self.model.eval()

        cnt = 0
        inputs = [d['sent'] for d in read_json(self.config.test_path)]
        sentence_ids = []
        new_pred_results = []
        golds = []
        # with open(outfile_txt, 'w') as fout:
        data = []
        # batch:5*8*512, 5是tokenizer有5个结果：input_ids, token_type_ids, attention_mask, offset_mapping和labels
        for batch in tqdm.tqdm(self.dataloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)

            inputs_ids, token_type_ids, attention_mask, tgt_inputs_ids, tgt_attention_mask, sent_ids = batch
            outputs = self.model.model.generate(input_ids=inputs_ids, attention_mask=attention_mask,
                                                num_beams=self.config.num_beams, max_length=350, use_cache=True,
                                                return_dict_in_generate=True, output_hidden_states=True,
                                                output_scores=True)
            # emissions = self.model(inputs_ids, token_type_ids, attention_mask)
            pred_ids = outputs[0]
            gold = self.tokenizer.batch_decode(tgt_inputs_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True)
            preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            pred_results = [p.replace(' ', '').replace('[SEP]', '') for p in preds]
            new_pred_results.extend(pred_results)

            golds += [g.replace(' ', '') for g in gold]

            sentence_ids += sent_ids.cpu().tolist()

        pred_datas = []
        pred_results = new_pred_results
        assert len(sentence_ids) == len(pred_results) == len(golds)
        for sentId, revisedSent in zip(sentence_ids, new_pred_results):
            pred_datas.append({
                "sent_id": sentId,
                "revisedSent": revisedSent
            })
        save_json(pred_datas, outfile_txt)
        # if not self.config.just_bart:
        #     macrof1, microf1 = New_multilabel_evaluate(classify_pred, classify_gold, verbose=True, id2labels=self.model.processor.label_schema.id2labels, error_only=self.config.error_only)

        char_f1, word_f1 = rewrite_eval(self.config, golds, pred_results, sentence_ids,
                                        save_path=outfile_txt.replace('.json', ".hyp.para"),
                                        data_path=self.config.test_path)
        bleu_score = bleu(pred_results, golds) * 100
        # levenshtein_tgt = levenshtein(pred_results,golds)
        levenshtein_input = levenshtein(pred_results, inputs)
        bert_s = bert_score(preds=pred_results, golds=golds)
        ppl = bert_ppl(pred_results)
        em_acc = em_eval(golds, pred_results)
        print(f"the EM score: {em_acc * 100:.2f}")

        print(f"char f0.5: {char_f1}")
        print(f"word_f0.5: {word_f1}")
        print(f"the bleu score: {bleu_score:.4f}")
        print(f"the bert score: {bert_s:.4f}")
        print(f"the bert ppl score: {ppl:.4f}")
        # print(f"the levenshtein score with gold seq: {levenshtein_tgt:.4f}")
        print(f"the levenshtein score with input seq: {levenshtein_input:.4f}")

        print('done--all %d tokens.' % cnt)

