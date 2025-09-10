import json
import os
import re
import random
from collections import Counter
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from clean_gadget import clean_gadget
from parse import tokenCNN_parse, GRU_parse


# ------------------------------------------------set parse---------------------------
def file_parse():
    parser = argparse.ArgumentParser(description='Normalization.')
    # parser.add_argument('-embed_size', type=int, default=13, help='嵌入的维度')

    # 路径参数
    parser.add_argument('-org_datapath', default='/home/ysusmart/xhz/data/CodeXGLUE/function.json',
                        help='The dir path of input dataset.', type=str)
    parser.add_argument('-normalized_data_path', default='/home/ysusmart/xhz/data/CodeXGLUE/norm_data.json',
                        help='The path for storing the normalized data.', type=str)
    parser.add_argument('-split_data_path', default='/home/ysusmart/xhz/data/CodeXGLUE/split_data',
                        help='The path for storing the normalized data.', type=str)
    parser.add_argument('-split_flag_path', default='/home/ysusmart/xhz/data/CodeXGLUE/split_flag',
                        help='The path for storing the normalized data.', type=str)
    parser.add_argument('-dict_path', default='/home/ysusmart/xhz/data/CodeXGLUE/dict.json',
                        help='The file path of dict data.', type=str)
    args = parser.parse_args()
    return args


def file_parse_noNorm():
    parser = argparse.ArgumentParser(description='Normalization.')
    # parser.add_argument('-embed_size', type=int, default=13, help='嵌入的维度')

    # 路径参数
    parser.add_argument('-org_datapath', default='/home/ysusmart/xhz/data/CodeXGLUEnoNorm/function.json',
                        help='The dir path of input dataset.', type=str)
    parser.add_argument('-normalized_data_path', default='/home/ysusmart/xhz/data/CodeXGLUEnoNorm/norm_data.json',
                        help='The path for storing the normalized data.', type=str)
    parser.add_argument('-split_data_path', default='/home/ysusmart/xhz/data/CodeXGLUEnoNorm/split_data',
                        help='The path for storing the normalized data.', type=str)
    parser.add_argument('-split_flag_path', default='/home/ysusmart/xhz/data/CodeXGLUEnoNorm/split_flag',
                        help='The path for storing the normalized data.', type=str)
    parser.add_argument('-dict_path', default='/home/ysusmart/xhz/data/CodeXGLUEnoNorm/dict.json',
                        help='The file path of dict data.', type=str)
    args = parser.parse_args()
    return args


# ------------------------------------------------normalized---------------------------
def normalize(args):
    result = []

    # 获取文件夹 （有缺陷和无缺陷文件夹）
    with open(args.org_datapath, "r") as f:
        content = json.load(f)
        # bad = content[492]
        for item in content:
            # print(item['target'])
            result.append(pro_one_item(item))

            # pro_one_item(item, args)

    # 写回归一化的数据
    with open(args.normalized_data_path, "w") as f:
        json_data = json.dumps(result)
        f.write(json_data)


def pro_one_item(item):
    # 格式化标签
    if item['target'] == 0:
        tag = [int(1), int(0)]
    else:
        tag = [int(0), int(1)]
    code = item['func']
    result = []
    retraction = []

    # 删除注释
    pattern = r'(?<!:)//.*|/\*.*?\*/'
    code = re.sub(pattern, '', code)
    code = re.sub(pattern, '', code, flags=re.DOTALL)
    # code = re.sub('(?<!:)\/\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
    code = code.splitlines()
    # nor_code = clean_gadget(code)

    # 删除空行00
    for line in code:
        pattern = r'^[\n\s]+$'
        if re.match(pattern, line) or line == '':
            continue
        else:
            retraction.append(len(line) - len(line.lstrip()))
            result.append(split_str(line))
            # result.append(line.lstrip())
    # 构建字典进行打包
    data_dict = {
        'code': result,
        'retraction': norm_retraction(retraction),
        'tag': tag
    }
    return data_dict


# 方便后面划词使用
def split_str(str):
    str = str.replace('\t', '    ')
    str = str.replace('(', " ( ")
    str = str.replace(')', " ) ")
    str = str.replace('{', " { ")
    str = str.replace('}', " } ")
    str = str.replace(':', " : ")
    str = str.replace(';', " ; ")
    str = str.replace(',', " , ")
    str = str.replace("'", " ' ")
    str = str.replace('"', ' " ')
    str = str.replace('+', " + ")
    str = str.replace('-', " - ")
    str = str.replace('_', " _ ")
    str = str.replace('*', " * ")
    str = str.replace('/', " / ")
    str = str.replace('%', " % ")
    str = str.replace('->', " -> ")
    str = str.replace('!', " ! ")
    str = str.replace('>', " > ")
    str = str.replace('<', " < ")
    str = str.replace('=', " = ")
    str = str.replace('[', " [ ")
    str = str.replace(']', " ] ")
    str = str.replace('&', " & ")
    str = str.replace('.', " . ")
    str = str.replace('~', " ~ ")
    str = str.replace('|', " | ")
    str = str.replace('?', " ? ")
    str = str.replace('^', " ^ ")
    str = str.replace('\n', " \n ")
    str = str.replace('\\', " \\ ")
    return str.lstrip().split()


# 归一化缩进
def norm_retraction(retraction_list):
    result = []
    if 3 in retraction_list:
        for num in retraction_list:
            result.append(int(num / 3))
    elif 4 in retraction_list:
        for num in retraction_list:
            result.append(int(num / 4))
    else:
        for num in retraction_list:
            result.append(int(num / 3))
    return result


# --------------------------------Building Dict ------------------------------------
def GetDict(counter_dict, save_path):
    dict = {'<p>': 0}  # 添加特殊字符字典
    i = int(1)
    for vocab in counter_dict:
        dict.update({vocab: i})
        i += 1
    print("字典长度：{}".format(len(dict)))
    with open(save_path, 'w') as f:
        json_data = json.dumps(dict)
        f.write(json_data)


def BuildVocab(args):
    counter = Counter()
    with open(args.normalized_data_path, "r") as f:
        data = json.load(f)

    for item in data:
        code = item['code']
        for line in code:
            counter.update(line)

    sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # vocabs = vocab(counter, specials=['<p>'], special_first=True)
    # print(vocabs.get_stoi())
    # print(len(sorted_counter))
    counter_dict = dict(sorted_counter)
    GetDict(counter_dict, args.dict_path)


# --------------------------------split dataset--------------------------------
def a_split(data_path, flag_path, save_path):
    # 在此根据路径获取指定标签的数据 调用三次得到三个数据集
    result = []
    with open(data_path, "r") as f:
        data = json.load(f)

    with open(flag_path, "r") as f:  # 打开文件
        indexes = f.readlines()  # 读取文件

    for index in indexes:
        result.append(data[int(index.strip('\n'))])

    with open(save_path, 'w') as f:
        json_data = json.dumps(result)
        f.write(json_data)


def split_dataset(args):
    a_split(args.normalized_data_path, args.split_flag_path + "//test.txt", args.split_data_path + "//test.json")
    a_split(args.normalized_data_path, args.split_flag_path + "//valid.txt", args.split_data_path + "//valid.json")
    a_split(args.normalized_data_path, args.split_flag_path + "//train.txt", args.split_data_path + "//train.json")


# --------------------------------statistics--------------------------------
def statistical_dataset(args):
    positive_sample = 0
    negative_sample = 0

    code_overflow_num = 0
    line_overflow_num = 0

    code_overflow_flag = 64
    line_overflow_flag = 32

    line_len_list = []
    code_len_list = []
    retraction_num_list = []

    with open(args.dict_path, "r") as f:
        dicts = json.load(f)
        print("字典长度:{}".format(len(dicts)))

    test_path = args.split_data_path + "//test.json"
    train_path = args.split_data_path + "//train.json"
    valid_path = args.split_data_path + "//valid.json"
    with open(test_path, "r") as f:
        dataset = json.load(f)
        leng = len(dataset)
        vul = 0
        for data in dataset:
            if data['tag'][0] == 0:
                vul += 1
        print("测试集缺陷比{}".format(vul / leng))
        print("测试集数据量{}".format(leng))

    with open(train_path, "r") as f:
        dataset = json.load(f)
        leng = len(dataset)
        vul = 0
        for data in dataset:
            if data['tag'][0] == 0:
                vul += 1
        print("训练集缺陷比{}".format(vul / leng))
        print("训练集数据量{}".format(leng))
    with open(valid_path, "r") as f:
        dataset = json.load(f)
        leng = len(dataset)
        vul = 0
        for data in dataset:
            if data['tag'][0] == 0:
                vul += 1
        print("验证集缺陷比{}".format(vul / leng))
        print("验证集数据量{}".format(leng))

    with open(args.normalized_data_path, "r") as f:
        dataset = json.load(f)

    # 统计数据
    for data in dataset:
        # np.append(retraction_num_list, np.array(data['retraction']))
        retraction_num_list.extend(data['retraction'])
        code = data['code']
        tag = data['tag']
        # np.append(code_len_list, len(code))
        code_len_list.append(len(code))
        if len(code) > code_overflow_flag:
            code_overflow_num += 1
            # print(args.split_data_path + "//" + setfolder + "//" + filename)
        for line in code:
            # np.append(line_len_list, len(line))
            line_len_list.append(len(line))
            if len(line) > line_overflow_flag:
                line_overflow_num += 1

        if tag[0] == 1:
            negative_sample += 1
        elif tag[0] == 0:
            positive_sample += 1
        else:
            print("警告：标签中含有意料之外的数字")
    line_len_list = np.array(line_len_list)
    code_len_list = np.array(code_len_list)
    retraction_num_list = np.array(retraction_num_list)
    print("代码行最大长度:{}".format(np.max(line_len_list)))
    print("代码行平均长度:{}".format(np.average(line_len_list)))
    print("代码段最大长度:{}".format(np.max(code_len_list)))
    print("代码段平均长度:{}".format(np.average(code_len_list)))
    print("缩进标签最大值:{}".format(np.max(retraction_num_list)))
    print("负样本数量:{}".format(negative_sample))
    print("负样本占比:{}".format(negative_sample / (negative_sample + positive_sample)))
    print("正样本数量:{}".format(positive_sample))
    print("正样本占比:{}".format(positive_sample / (negative_sample + positive_sample)))
    print("代码段截断溢出占比>{}".format(code_overflow_flag),
          ":{}".format(code_overflow_num / (negative_sample + positive_sample)))
    print("代码行截断溢出占比>{}".format(line_overflow_flag), ":{}".format(line_overflow_num / (np.sum(code_len_list))))


# --------------------------------Get Best Model--------------------------------
def is_best(test_dict, best_dict):
    best_flag = 0
    metrics = ['roc', 'acc', 'precision', 'recall', 'f1']
    for i in range(len(metrics)):
        if test_dict[metrics[i]] > best_dict[metrics[i]]:
            best_flag += 1
            if metrics[i] == 'acc':
                best_flag += 1
    if test_dict['score'] > best_dict['score']:
        best_flag += 1
    if best_flag >= 3:
        return True
    else:
        return False


def getscore(metric_dict):
    score = 0
    metrics = ['acc', 'precision', 'recall', 'f1']
    rank_list = {
        'acc': [59.37, 60.69, 61.05, 62.48, 63.36, 62.81, 63.84, 64.09, 59.99, 59.40, 61.71, 60.05, 64.46],
        'precision': [65.49, 61.75, 62.10, 63.00, 57.01, 59.10, 64.41, 59.79, 61.87],
        'recall': [42.79, 50.03, 54.58, 52.91, 52.51, 37.74, 37.21, 39.92, 58.99],
        'f1': [51.76, 55.28, 58.10, 57.51, 54.67, 46.07, 47.17, 47.87, 60.39]
    }
    for metric in metrics:
        for item in rank_list[metric]:
            if metric_dict[metric] * 100 >= item:
                score += 1
    metric_dict['score'] = score
    # metric_dict['score'] = 0
    return metric_dict


# --------------------------------Build DataSet--------------------------------
class MyDataset(Dataset):
    def __init__(self, data_path, dict_path, parser, dim=1, need_clean=False, no_code_len=False,
                 need_slice=False):  # 初始化
        # 这个parser是模型的参数而不是通用参数
        self.no_code_len = no_code_len
        with open(dict_path, "r") as f:
            self.dicts = json.load(f)
        with open(data_path, "rb") as file:
            self.data = json.load(file)
        self.parser = parser
        self.dim = dim
        self.slice = need_slice
        if need_slice:
            need_clean = True

        if dim == 2 and need_clean == True and need_slice == False:
            self.cleaned_index, __ = self.clean_static()
        elif dim == 2 and need_clean == True and need_slice == True:
            __, self.cleaned_index = self.clean_static()
        else:
            self.cleaned_index = range(len(self.data))
        self.len = len(self.cleaned_index)

    def __getitem__(self, index):
        # data_lists, tags, retraction = RebuildHdf5Data(self.data_path) # 放在初始化里防止重复计算
        # self.len = len(self.tags)
        item = self.data[self.cleaned_index[index]]
        if self.dim == 1 and self.no_code_len == False:
            code = self.rebuild_data_onedim(item['code'])
            # retraction = self.rebuild_retraction(item['retraction'])
            return code, torch.Tensor(item['tag'])
        if self.dim == 2 and self.no_code_len == False and self.slice == False:
            code = self.rebuild_data_twodim(item['code'])
            retraction = self.rebuild_retraction(item['retraction'])
            return code, torch.Tensor(item['tag']), retraction
        # 此情况为需要切片的情况
        if self.dim == 2 and self.no_code_len == False and self.slice == True:
            code, slice_num = self.rebuild_data_twodim_slice(item['code'])
            retraction = self.rebuild_retraction(item['retraction'])
            return code, torch.Tensor(item['tag']), retraction, slice_num

        if self.dim == 1 and self.no_code_len == True:  # error need max len
            code = self.rebuild_data_onedim_nocodelen(item['code'])
            return code, torch.Tensor(item['tag'])

    def __len__(self):
        return self.len

    def clean_static(self):
        # 统计离群数据，并返回一个数组标注哪些数据是留下来的
        data_num = len(self.data)
        del_flag = np.zeros(data_num, dtype=int)
        cleaned_data_index = []
        del_data_index = []
        for i in range(data_num):
            item = self.data[i]
            code = item['code']
            if len(code) > self.parser.code_len * 1.5:
                del_flag[i] = 1
                continue
            # 这样删的太多了
            # for line in code:
            # if len(line) > self.parser.line_len * 1.5:
            # del_flag[i] = 1
            # break

        for i in range(data_num):
            if del_flag[i] == 0:
                cleaned_data_index.append(i)
            elif del_flag[i] == 1:
                del_data_index.append(i)
        print("删除离群数据{}条".format(data_num - len(cleaned_data_index)),
              "删除离群数据占比:{}".format((data_num - len(cleaned_data_index)) / data_num),
              "还剩{}条数据".format(len(cleaned_data_index)))
        return cleaned_data_index, del_data_index

    def rebuild_data_onedim(self, code):
        data = np.zeros(self.parser.code_len, dtype=int)
        data_length = 0
        for line in code:
            if data_length >= self.parser.code_len:
                break
            for vocab in line:
                if data_length >= self.parser.code_len:
                    break
                if vocab is not None:
                    data[data_length] = self.dicts[vocab]
                    data_length += 1
        return data

    def rebuild_data_onedim_nocodelen(self, code):  # error
        data = []
        for line in code:
            for vocab in line:
                if vocab is not None:
                    data.append(self.dicts[vocab])
        return np.array(data)

    def rebuild_data_twodim_slice(self, code):
        data = np.zeros((self.parser.code_len * self.parser.max_slice_num, self.parser.line_len), dtype=int)
        slice_num = min(int(len(code) / self.parser.code_len), self.parser.max_slice_num)

        for i in range(min(self.parser.code_len * self.parser.max_slice_num, len(code))):
            line = code[i]
            for j in range(min(self.parser.line_len, len(line))):
                data[i, j] = self.dicts[line[j]]
        # 构建切片表达
        sliced_data = data.reshape(self.parser.max_slice_num, self.parser.code_len, self.parser.line_len)
        return sliced_data, slice_num

    def rebuild_data_twodim(self, code):
        data = np.zeros((self.parser.code_len, self.parser.line_len), dtype=int)
        for i in range(min(self.parser.code_len, len(code))):
            line = code[i]
            for j in range(min(self.parser.line_len, len(line))):
                data[i, j] = self.dicts[line[j]]
        return data

    def rebuild_retraction(self, retraction):
        data = np.zeros(self.parser.code_len, dtype=int)
        data_length = 0
        for val in retraction:
            if data_length >= self.parser.code_len:
                break
            if val is not None:
                data[data_length] = val
                data_length += 1
        return data


def main():
    # 读入参数  即文件路径
    args = file_parse_noNorm()
    # 标准化过程
    normalize(args)
    print("-----标准化完成-----")
    # 构建字典
    BuildVocab(args)
    print("-----构建字典完成-----")
    split_dataset(args)
    print("-----数据集划分完成-----")
    print("-----开始统计数据-----")
    statistical_dataset(args)
    # model_args = tokenCNN_parse()
    # test_data = OneDimDataset(args.split_data_path + "//test.json", args.dict_path, model_args)
    # merged_test = DataLoader(dataset=test_data, batch_size=model_args.batch_size, shuffle=True, drop_last=True)
    # for batch in merged_test:
    #  data, tag = batch
    # print(data)
    # print(tag)


if __name__ == '__main__':
    main()
