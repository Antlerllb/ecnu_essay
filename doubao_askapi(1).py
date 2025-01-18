import re
import os,sys
sys.append(os.path.dirname(os.path.dirname(os.absname(__file__))))

from openai import OpenAI
from volcenginesdkarkruntime import Ark #豆包的下载的包

def doubao1(input_content: str = None):
    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    # Non-streaming:
    print("----- standard request -----")
    # print(type(input_content))
    #这里插入prompt
    prompt = f"""请专注剖析以下句子，精准判别其中是否存在如下语法错误类别：
        缺字漏字、错别字错误、缺少标点、错用标点、主语不明、谓语残缺、宾语残缺、
        其他成分残缺、主语多余、虚词多余、其他成分多余、语序不当、动宾搭配不当、
        其他搭配不当或无错误。待分析句子为：{input_content}。若存在错误，
        以最小化修正原则给出改正后的句子，并采用 JSON 格式返回错误类别；
        若无错误，则径直返回"无错误"。输出需简洁明晰，无冗余表述。示例如下：
        输入：我过了一会儿，我完成了测试。
        输出：'{"错误类型":"主语多余","纠正句子":"过了一会儿，我完成了测试。"}'
        输入：有人认为富足的物质条件有利于孩子成长。
        输出：'{"错误类型":"无错误","纠正句子":""}'"""
    
    # base_prompt = "请专注剖析以下句子，精准判别其中是否存在如下语法错误类别："
    # error_categories = "缺字漏字、错别字错误、缺少标点、错用标点、主语不明、谓语残缺、宾语残缺、其他成分残缺、主语多余、虚词多余、其他成分多余、语序不当、动宾搭配不当、其他搭配不当或无错误。"
    # instruction = "若存在错误，以最小化修正原则给出改正后的句子，并采用 JSON 格式返回错误类别；若无错误，则径直返回“无错误”。请注意，这里的都是繁体作文，修改句子需要以香港的地区习俗文化出发，在繁体中文中，单引号通常使用「」，双引号使用『』。繁体语境当中，语言的使用也存在简繁混杂的情况，不要刻意将输入的句子当中的简体字转为繁体输出。输出需简洁明晰，无冗余表述。示例如下："
    # example1 = "输入：我过了一会儿，我完成了测试。输出：{'错误类型': '主语多余', '纠正句子': '过了一会儿，我完成了测试。'}"
    # example2 = "输入：有人认为富足的物质条件有利于孩子成长。输出：{'错误类型': '无错误', '纠正句子': ''}"
    # prompt = base_prompt + error_categories + "待分析句子为：" + input_content + "." + instruction + example1 + example2
    completion = client.chat.completions.create(
        model="ep-20241204225255-dp9xk",#指定模型的类型
        
        messages = [
            {"role": "system", "content": "你是香港的高级语文老师"},
            {"role": "user", "content": prompt },#添加prompt
        ],
    )
    return completion.choices[0].message.content

result = doubao1("今天天气真好，我和朋友一起出去玩")

