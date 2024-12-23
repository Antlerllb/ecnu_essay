import json
import re
import opencc

from utils import get_path, save_json


def split_sentences(text):
    # 使用正则表达式匹配句子结束符，包括中文句号、感叹号、问号等
    sentence_endings = re.compile(r'([。！？\?])([^”’」』])')
    text = sentence_endings.sub(r"\1\n\2", text)

    # 处理句尾符号后跟引号的情况
    text = re.sub(r'(\.{6})([^”’」』])', r'\1\n\2', text)  # 半角省略号
    text = re.sub(r'(\…{2})([^”’」』])', r'\1\n\2', text)  # 全角省略号
    text = re.sub(r'([。！？\?][”’」』])([^，。！？\?])', r'\1\n\2', text)

    # 将文本按行分割成句子
    sentences = text.strip().split("\n")

    # 去除可能的空行并返回句子列表
    return [sentence.strip() for sentence in sentences if sentence.strip()]

converter = opencc.OpenCC('t2s.json')
sents_biaozhu = json.load(open(get_path('1205_sentence_biaozhu.json')))
for biaozhu_item in sents_biaozhu:
    biaozhu_item['sc'] = converter.convert(biaozhu_item['sentence'])
sents_in_split = [j['sc'] for j in sents_biaozhu]
sents_in_raw = []
with open(get_path('datas/lijiang_essays.jsonl')) as f:
    for lijiang_line in f.readlines():
        lijiang_json = json.loads(lijiang_line.strip())
        essay = lijiang_json['essay']
        lijiang_sents = split_sentences(essay)
        for lijiang_s in lijiang_sents:
            lijiang_sc = converter.convert(lijiang_s)
            if lijiang_sc in sents_in_split:
                for idx, biaozhu_item in enumerate(sents_biaozhu):
                    if lijiang_sc == biaozhu_item['sc']:
                        sents_biaozhu[idx]['flag'] = True
                        sents_biaozhu[idx]['essay_id'] = lijiang_json['id']
                        sents_biaozhu[idx]['sent'] = lijiang_s
                        sents_biaozhu[idx]['grade'] = lijiang_json['grade']
                        sents_biaozhu[idx]['sc_sent'] = converter.convert(lijiang_s)

new_json_data = []
not_equal_num = 0
for biaozhu_item in sents_biaozhu:
    if biaozhu_item.get('flag'):
        new_json_data.append({
            'sent_id': biaozhu_item['id'],
            'essay_id': biaozhu_item['essay_id'],
            'grade': biaozhu_item['grade'],
            'sent': biaozhu_item['sent'],
            'sc_sent': biaozhu_item['sc_sent'],
            'fine_grained_error_type': [biaozhu_item['error_type']],
            'revised_sent': biaozhu_item['revised_sent'],
        })
# save_json(new_json_data, get_path('lijiang_sents_20241218.json'))

"""
{  
  "sent_id": 9341,
  "essay_id": 1,
  "sent": "我還要感謝我的同學，回憶著我的初中時光，我的腦海總映放著各位同學與我一起學習、進步、相互幫助的畫面，大家總爭著第一，早早的踏入校園，用行動書寫著「一年之計在於晨」。", 
  "sc_sent": "我还要感谢我的同学，回忆着我的初中时光，我的脑海总映放着各位同学与我一起学习、进步、相互帮助的画面，大家总争着第一，早早的踏入校园，用行动书写着“一年之计在于晨”。",
  "course_grained_error_type": [  
    "字符级错误",  
    "成分搭配不当型错误"  
  ],  
  "fine_grained_error_type": [  
    "错别字错误",  
    "语序不当"  
  ],
  "revised_sent": "我還要感謝我的同學，回憶著我的初中時光，我的腦海總放映著各位同學與我一起學習、進步、相互幫助的畫面，大家總爭著第一，早早地踏入校園，用行動書寫著「一年之計在於晨」。"
}

"""