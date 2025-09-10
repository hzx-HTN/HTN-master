# coding=utf-8
import os
import re
import shutil
import pickle
from collections import Counter
import numpy as np
import random
from clean_gadget import clean_gadget
from parse import generic_parse

data_split_flag = 0


# ------------------------------------------------normalized---------------------------
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


def normalize(args):
    # 获取文件夹 （有缺陷和无缺陷文件夹）
    setfolderlist = os.listdir(args.org_datapath)
    for setfolder in setfolderlist:
        # 获取文件夹中的数据数据名
        catefolderlist = os.listdir(args.org_datapath + "//" + setfolder)
        # print(catefolderlist)

        # 单个文件归一化流程
        for filename in catefolderlist:
            if filename == "retraction" or filename == "result":
                continue
            filepath = args.org_datapath + "//" + setfolder
            # print(catefolder)

            # 归一化一个文件
            pro_one_file(filepath, filename, args)


def pro_one_file(filepath, filename, args):
    # 判断属于哪个文件夹 得到标签
    if "Non_vulnerable_functions" in filepath or "No-Vul" in filepath:
        tag = [1, 0]
    else:
        tag = [0, 1]
    datapath = filepath + "//" + filename
    with open(datapath, "r") as file:
        code = file.read()
    file.close()
    result = []
    retraction = []
    # 删除注释
    code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
    # 写回
    with open(datapath, "w") as file:
        file.write(code.strip())
    file.close()

    # 按行处理
    with open(datapath, "r") as file:
        # 按行读取
        org_code = file.readlines()
        # print(org_code)
        # 处理程序
        nor_code = clean_gadget(org_code)
    file.close()

    # 删除空行
    for line in nor_code:
        pattern = r'^[\n\s]+$'
        if re.match(pattern, line):
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
    if tag[0] == 1:
        with open(args.normalized_data_path + "//" + "Vul" + "//" + filename + ".pkl", "wb") as file:
            pickle.dump(data_dict, file)
        file.close()
    elif tag[0] == 0:
        with open(args.normalized_data_path + "//" + "No_Vul" + "//" + filename + ".pkl", "wb") as file:
            pickle.dump(data_dict, file)
    else:
        print("警告（pro_one_file）：意料之外的标签！")


# --------------------------------Building Dict ------------------------------------
def GetDict(counter_dict, save_path):
    dict = {'<p>': 0}  # 添加特殊字符字典
    i = int(1)
    for vocab in counter_dict:
        dict.update({vocab: i})
        i += 1
    # print(len(dict))
    with open(save_path, "wb") as file:
        # 写
        pickle.dump(dict, file)
    file.close()


def BuildVocab(args):
    counter = Counter()
    filepath = args.normalized_data_path
    foldlist = os.listdir(filepath)
    for fold in foldlist:
        filelist = os.listdir(filepath + "//" + fold)
        for filename in filelist:
            with open(filepath + "//" + fold + "//" + filename, "rb") as file:
                # 读
                data = pickle.load(file)
                code = data['code']
            file.close()
            for line in code:
                counter.update(line)

    sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # vocabs = vocab(counter, specials=['<p>'], special_first=True)
    # print(vocabs.get_stoi())
    # print(len(sorted_counter))
    counter_dict = dict(sorted_counter)
    GetDict(counter_dict, args.dict_path)


# --------------------------------split dataset--------------------------------
def save_dataset(org_path, filelist, savepath):
    for filename in filelist:
        with open(org_path + "//" + filename, "rb") as file:
            data = pickle.load(file)
        file.close()
        with open(savepath + "//" + filename, "wb") as file:
            pickle.dump(data, file)
        file.close()


def split_dataset(args):
    split_flag = 0
    norm_path = args.normalized_data_path
    split_path = args.split_data_path
    filelist = os.listdir(norm_path)

    # 在有缺陷和无缺陷两个文件夹 分别随机按118比例划分

    foldlist = os.listdir(norm_path)
    for fold in foldlist:
        filelist = os.listdir(norm_path + "//" + fold)
        sample_size = int(len(filelist) * 0.1)
        test_filelist = random.sample(filelist, sample_size)
        filelist = [s for s in filelist if s not in test_filelist]
        validate_list = random.sample(filelist, sample_size)
        train_filelist = [s for s in filelist if s not in validate_list]

        save_dataset(norm_path + "//" + fold, test_filelist, split_path + "//" + "test")
        save_dataset(norm_path + "//" + fold, train_filelist, split_path + "//" + "train")
        save_dataset(norm_path + "//" + fold, validate_list, split_path + "//" + "validate")


# --------------------------------statistics--------------------------------
def statistical_dataset(args):
    positive_sample = 0
    negative_sample = 0

    code_overflow_num = 0
    line_overflow_num = 0

    line_len_list = []
    code_len_list = []
    retraction_num_list = []

    with open(args.dict_path, "rb") as file:
        dicts = pickle.load(file)
        print("字典长度:{}".format(len(dicts)))
    file.close()

    test_path = args.split_data_path + "//test"
    train_path = args.split_data_path + "//train"
    validate_path = args.split_data_path + "//validate"

    test_filelist = os.listdir(test_path)
    train_filelist = os.listdir(train_path)
    validate_filelist = os.listdir(validate_path)

    print("训练集数据条数:{}".format(len(test_filelist)))
    print("测试集数据条数:{}".format(len(train_filelist)))
    print("验证集数据条数:{}".format(len(validate_filelist)))

    # 统计数据
    setfolderlist = os.listdir(args.split_data_path)
    for setfolder in setfolderlist:
        catefolderlist = os.listdir(args.split_data_path + "//" + setfolder)
        for filename in catefolderlist:
            with open(args.split_data_path + "//" + setfolder + "//" + filename, "rb") as file:
                data = pickle.load(file)
            file.close()
            # np.append(retraction_num_list, np.array(data['retraction']))
            retraction_num_list.extend(data['retraction'])
            code = data['code']
            tag = data['tag']
            # np.append(code_len_list, len(code))
            code_len_list.append(len(code))
            if len(code) > 64:
                code_overflow_num += 1
                # print(args.split_data_path + "//" + setfolder + "//" + filename)
            for line in code:
                # np.append(line_len_list, len(line))
                line_len_list.append(len(line))
                if len(line) > 16:
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
    print("代码段溢出占比:{}".format(code_overflow_num / (negative_sample + positive_sample)))
    print("代码行溢出占比:{}".format(line_overflow_num / (np.sum(code_len_list))))


def main():
    # 读入参数  即文件路径
    args = generic_parse()
    # 标准化过程
    normalize(args)
    print("-----标准化完成-----")
    # 构建字典
    # BuildVocab(args)
    print("-----构建字典完成-----")
    # split_dataset(args)
    print("-----数据集划分完成-----")
    print("-----开始统计数据-----")
    statistical_dataset(args)


if __name__ == '__main__':
    main()
