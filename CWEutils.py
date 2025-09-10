import json
import os
import re
import random
from collections import Counter
import argparse
import h5py
from CodeXGLUEutils import norm_retraction, split_str, GetDict, statistical_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from clean_gadget import clean_gadget
from parse import tokenCNN_parse, GRU_parse


# ------------------------------------------------set parse---------------------------
def CWEfile_parse():
    parser = argparse.ArgumentParser(description='Normalization.')
    # parser.add_argument('-embed_size', type=int, default=13, help='嵌入的维度')
    # 路径参数
    parser.add_argument('-datapath', default='/home/ysusmart/xhz/data/CWEdataset',
                        help='The dir path of input dataset.', type=str)
    parser.add_argument('-split_data_path', default='/home/ysusmart/xhz/data/CWEdataset',
                        help='The dir path of input dataset.', type=str)
    parser.add_argument('-normalized_data_path', default='/home/ysusmart/xhz/data/CWEdataset/norm_merged_data.json',
                        help='The dir path of input dataset.', type=str)
    parser.add_argument('-dict_path', default='/home/ysusmart/xhz/data/CWEdataset/dict.json',
                        help='The file path of dict data.', type=str)
    args = parser.parse_args()
    return args


def CWEBalancefile_parse():
    parser = argparse.ArgumentParser(description='Normalization.')
    # parser.add_argument('-embed_size', type=int, default=13, help='嵌入的维度')
    # 路径参数
    parser.add_argument('-datapath', default='/home/ysusmart/xhz/data/CWEdataset',
                        help='The dir path of input dataset.', type=str)
    parser.add_argument('-split_data_path', default='/home/ysusmart/xhz/data/CWE_BalancedDataset',
                        help='The dir path of input dataset.', type=str)
    parser.add_argument('-normalized_data_path',
                        default='/home/ysusmart/xhz/data/CWE_BalancedDataset/norm_merged_data.json',
                        help='The dir path of input dataset.', type=str)
    parser.add_argument('-dict_path', default='/home/ysusmart/xhz/data/CWE_BalancedDataset/dict.json',
                        help='The file path of dict data.', type=str)
    args = parser.parse_args()
    return args


# ------------------------------------------------normalized---------------------------
def pro_one_file(args, filename, savename):
    result = []
    CWEdict = ['CWE-119', 'CWE-120', 'CWE-469', 'CWE-476', 'CWE-other']
    dataname = 'functionSource'
    f = h5py.File(args.datapath + '/' + filename, 'r')
    data = f[dataname]

    for i in range(len(data)):
        tag = []
        for vul_type in CWEdict:
            tag.append(f[vul_type][i])
        result.append(pro_one_item(data[i].decode('utf-8'), tag))

    with open(args.datapath + '/' + savename, "w") as f:
        json_data = json.dumps(result)
        f.write(json_data)


def pro_one_item(code, tag_list):
    # 格式化标签
    if 1 in tag_list:
        tag = [int(0), int(1)]
    else:
        tag = [int(1), int(0)]

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


def normalize(args):
    pro_one_file(args, "test.hdf5", "test.json")
    print("Norm test finish!")
    pro_one_file(args, "validate.hdf5", "valid.json")
    print("Norm validate finish!")
    pro_one_file(args, "train.hdf5", "train.json")
    print("Norm train finish!")


# --------------------------------Building Dict ------------------------------------
def get_counter(args, filename, counter):
    with open(args.split_data_path + '/' + filename, "r") as f:
        data = json.load(f)

    for item in data:
        code = item['code']
        for line in code:
            counter.update(line)
    return counter


def BuildVocab(args):
    counter = Counter()

    counter = get_counter(args, 'test.json', counter)
    counter = get_counter(args, 'valid.json', counter)
    counter = get_counter(args, 'train.json', counter)

    sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    counter_dict = dict(sorted_counter)
    GetDict(counter_dict, args.dict_path)


# --------------------------------Merge Data ------------------------------------
def append_data(args, data_list, filename):
    with open(args.split_data_path + '/' + filename, "r") as f:
        data = json.load(f)
    for item in data:
        data_list.append(item)
    return data_list


def MergeData(args):
    data_list = []
    data_list = append_data(args, data_list, 'test.json')
    data_list = append_data(args, data_list, 'valid.json')
    data_list = append_data(args, data_list, 'train.json')
    with open(args.normalized_data_path, "w") as f:
        json_data = json.dumps(data_list)
        f.write(json_data)


# --------------------------------Balance Data ------------------------------------
# 只平衡训练集即可
def BalanceData(max_line_len, max_code_len, args, filename, rate):
    with open(args.datapath + '/' + filename, "r") as f:
        data = json.load(f)

    org_data_num = len(data)
    # 统计缺陷条数与非缺陷条数
    vul_num = static_vul_num(data)
    no_vul_num = int(vul_num / rate) - vul_num

    del_num = org_data_num - vul_num - no_vul_num

    # 统计无缺陷中溢出的数据
    # overflow_data_index中的数据是要删除的
    overflow_data_index, save_data_index = clean_static(data, max_line_len, max_code_len)
    overflow_num = len(overflow_data_index)

    del_data_index = random.sample(save_data_index, del_num - overflow_num)
    del_data_index = del_data_index + overflow_data_index
    # del_data_index即为所有需要丢弃元素的索引
    final_data_list = []
    del_flag = np.zeros(org_data_num, dtype=int)
    for index in del_data_index:
        del_flag[index] = 1
    del_sum = sum(del_flag)
    for i in range(org_data_num):
        item = data[i]
        if del_flag[i] == 0:
            final_data_list.append(item)

    with open(args.split_data_path + '/' + filename, "w") as f:
        json_data = json.dumps(final_data_list)
        f.write(json_data)


def static_vul_num(data):
    vul_num = 0
    for item in data:
        tag = item['tag']
        if tag[1] == 1:
            vul_num += 1
    return vul_num


def clean_static(data, max_line_len, max_code_len):
    # 统计离群数据，并返回一个数组标注哪些数据是留下来的
    data_num = len(data)
    del_flag = np.zeros(data_num, dtype=int)
    del_data_index = []
    save_data_index = []
    for i in range(data_num):
        item = data[i]
        code = item['code']
        if len(code) > max_code_len:
            del_flag[i] = 1
            continue
        for line in code:
            if len(line) > max_line_len:
                del_flag[i] = 1
            break
    for i in range(data_num):
        if del_flag[i] == 1 and data[i]['tag'][0] == 1:
            del_data_index.append(i)
        elif data[i]['tag'][0] == 1:
            # 缺陷中没删的
            save_data_index.append(i)
    return del_data_index, save_data_index


# --------------------------------Build Token ------------------------------------
# 一维和二维都要构建
class BuildToken():
    def __init__(self, data_path, dict_path, parser, dim=1, need_clean=False, no_code_len=False,
                 need_slice=False, need_org_code_str = False):  # 初始化
        # 这个parser是模型的参数而不是通用参数
        self.no_code_len = no_code_len
        with open(dict_path, "r") as f:
            self.dicts = json.load(f)
        with open(data_path, "rb") as file:
            self.data = json.load(file)
        self.parser = parser
        self.dim = dim
        self.slice = need_slice
        self.need_org_code_str = need_org_code_str
        if need_slice:
            need_clean = True

        if dim == 2 and need_clean == True and need_slice == False:
            self.cleaned_index, __ = self.clean_static()
        elif dim == 2 and need_clean == True and need_slice == True:
            __, self.cleaned_index = self.clean_static()
        else:
            self.cleaned_index = range(len(self.data))
        self.len = len(self.cleaned_index)

    def getitem(self, index):
        # data_lists, tags, retraction = RebuildHdf5Data(self.data_path) # 放在初始化里防止重复计算
        # self.len = len(self.tags)

        item = self.data[self.cleaned_index[index]]
        org_code_str = item['code']
        if(org_code_str is None):
            print("None Code")
        # if(self.need_org_code_str):
        #     print(org_code_str)
        if self.dim == 1 and self.no_code_len == False:
            code = self.rebuild_data_onedim(item['code'])
            # retraction = self.rebuild_retraction(item['retraction'])
            if self.need_org_code_str:
                return code, np.array(item['tag']),org_code_str
            else:
                return code, np.array(item['tag'])
        if self.dim == 2 and self.no_code_len == False and self.slice == False:
            code = self.rebuild_data_twodim(item['code'])
            retraction = self.rebuild_retraction(item['retraction'])
            if self.need_org_code_str:
                return code, np.array(item['tag']), retraction, org_code_str
            else:
                return code, np.array(item['tag']), retraction

        # 此情况为需要切片的情况
        if self.dim == 2 and self.no_code_len == False and self.slice == True:
            code, slice_num = self.rebuild_data_twodim_slice(item['code'])
            retraction = self.rebuild_retraction(item['retraction'])
            if self.need_org_code_str:
                return code, np.array(item['tag']), retraction, slice_num, org_code_str
            else:
                return code, np.array(item['tag']), retraction, slice_num

        if self.dim == 1 and self.no_code_len == True:  # error need max len
            code = self.rebuild_data_onedim_nocodelen(item['code'])
            if self.need_org_code_str:
                return code, np.array(item['tag']),org_code_str, org_code_str
            else:
                return code, np.array(item['tag']),org_code_str

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

    def get_token(self):
        result = []
        for i in range(self.len):
            result.append(self.getitem(i))
        return result


class MyCWEDataset(Dataset):
    def __init__(self, data_path, dict_path, parser, dim=1, need_clean=False, no_code_len=False,
                 need_slice=False, need_rebuild=False, usePercentage = 1.0,need_org_str = False):  # 初始化

            print("构建数据集中:", data_path)
            Token = BuildToken(data_path, dict_path, parser, dim=dim, need_clean=need_clean, no_code_len=no_code_len,
                               need_slice=need_slice,need_org_code_str=need_org_str)

            self.data = Token.get_token()
            self.usePercentage = usePercentage

    def __len__(self):
        return int(len(self.data)*self.usePercentage)

    def __getitem__(self, index):
        return self.data[index]


def main():
    # 读入参数  即文件路径
    # args = CWEfile_parse()
    # 标准化过程
    # normalize(args)
    # print("-----标准化完成-----")
    # 构建字典
    # BuildVocab(args)
    # print("-----构建字典完成-----")
    # 融合分开的数据集
    # MergeData(args)
    # print("-----数据集融合完成-----")
    # print("-----开始统计数据-----")
    # statistical_dataset(args)
    # model_args = tokenCNN_parse()
    # test_data = OneDimDataset(args.split_data_path + "//test.json", args.dict_path, model_args)
    # merged_test = DataLoader(dataset=test_data, batch_size=model_args.batch_size, shuffle=True, drop_last=True)
    # for batch in merged_test:
    #  data, tag = batch
    # print(data)
    # print(tag)

    # ----------balance data
    args = CWEBalancefile_parse()
    BalanceData(32, 64, args, "train.json", 0.4)
    BalanceData(32, 64, args, "test.json", 0.4)
    BalanceData(32, 64, args, "valid.json", 0.4)
    print("平衡数据完毕")
    MergeData(args)
    print("融合数据完毕")
    BuildVocab(args)
    print("-----构建字典完成-----")
    print("-----开始统计数据-----")
    statistical_dataset(args)


if __name__ == '__main__':
    main()
