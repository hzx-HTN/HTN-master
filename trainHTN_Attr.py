import os
import time

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn import metrics

from model import HTN_Attr as MyModule
from parse import HTN_parse
from CodeXGLUEutils import MyDataset, file_parse_noNorm, is_best, getscore
from CWEutils import CWEfile_parse, CWEBalancefile_parse, MyCWEDataset

import json
import torch

def no_collate_fn(batch):
    """不进行collate操作，直接返回原始数据列表"""
    return batch

def line_array_to_list(line_attention_weights):
    line_array = []
    for attention_weight in line_attention_weights:
        a_line = []
        for array in attention_weight:
            a_line.append(array.tolist())
        line_array.append(a_line)
    return line_array

def hunk_array_to_list(hunk_attention_weights):
    hunk_array = []
    for attention_weight in hunk_attention_weights:
        hunk_array.append(attention_weight.tolist())
    return hunk_array


def getAttrJson(model: MyModule, model_parse, data_parse, save_path,rate = 1.0):
    print("开始保存json")
    validate_data = MyCWEDataset(path_parse.split_data_path + "//valid.json", data_parse.dict_path, model_parse, dim=2,
                             need_clean=False,usePercentage = rate,need_org_str=True)
    merged_validate = DataLoader(dataset=validate_data, batch_size=1, shuffle=True,
                                 drop_last=True,collate_fn=no_collate_fn)

    # 加载valid数据集 然后测试
    jsData = {
        'embed_size': model_parse.embed_size,
        'code_len': model_parse.code_len,
        'line_len': model_parse.line_len,
        'line_layer_num': model_parse.line_layer_num,
        'hunk_layer_num': model_parse.hunk_layer_num,
        'head_num': model_parse.head_num,
        'data': [],
    }

    data_list = []  # 重命名变量避免冲突

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    model.set_return_attention(True)
    with torch.no_grad():
        for batch in merged_validate:
            data = torch.from_numpy(np.expand_dims(batch[0][0], axis=0)).cuda()
            tag = torch.from_numpy(np.expand_dims(batch[0][1], axis=0)).cuda()
            ret = torch.from_numpy(np.expand_dims(batch[0][2], axis=0)).cuda()
            code = batch[0][3]
            # data, tag, ret,code = batch
            # if torch.cuda.is_available():
            #     data = data.cuda()
            #     tag = tag.cuda()
            #     ret = ret.cuda()
            opt, line_attention_weights, hunk_attention_weights = model(data, ret, True)

            # 代码行数 *  层数 *  注意力头数(取平均1)  *
            aData = {
                'code':code,
                'line_attention_weights': line_array_to_list(line_attention_weights),  # 转换为可序列化的格式
                'hunk_attention_weights': hunk_array_to_list(hunk_attention_weights),
            }
            data_list.append(aData)

    jsData['data'] = data_list

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(jsData, f, indent=4, ensure_ascii=False)

    print(f"注意力数据已保存到: {save_path}")
    return jsData


if __name__ == '__main__':
    path_parse = CWEBalancefile_parse()
    model_parse = HTN_parse()
    ###############初始化###############
    best_dict = {
        'roc': 0.0,
        'acc': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'score': 0,
        'epoc': 0
    }

    best_valid = {
        'roc': 0.0,
        'acc': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'score': 0,
        'epoc': 0
    }

    logfile = open('MyModelLog.txt', 'w')
    min_loss = 1.5
    stop_flag = 0
    lr_flag = 0
    ####################################
    ###############随机种子###############
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ###############模型声明###############
    module = MyModule(model_parse)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.ASGD(module.parameters(), lr=model_parse.learn_rate)
    optimizer = torch.optim.Adam(module.parameters(), lr=model_parse.learn_rate)
    ####################################

    # ###############测试json导出###############
    getAttrJson(module,model_parse,path_parse,"/home/ysusmart/xhz/HTN-master/test.json",rate = 0.001)
    # ######################################

    # ###############加载数据集###############
    test_data = MyCWEDataset(path_parse.split_data_path + "//test.json", path_parse.dict_path, model_parse, dim=2,
                             need_clean=False)
    train_data = MyCWEDataset(path_parse.split_data_path + "//train.json", path_parse.dict_path, model_parse, dim=2,
                              need_clean=False, usePercentage=0.1)
    validate_data = MyCWEDataset(path_parse.split_data_path + "//valid.json", path_parse.dict_path, model_parse, dim=2,
                                 need_clean=False)
    # ######################################

    
    ###############cuda化###############
    # 一机多卡设置
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,0'  # 设置所有可以使用的显卡，共计四块
    # device_ids = [0]  # 选中其中两块
    # module = nn.DataParallel(module, device_ids=device_ids)  # 并行使用两块
    # net = torch.nn.Dataparallel(model)  # 默认使用所有的device_ids
    module = module.cuda()
    loss_fn = loss_fn.cuda()
    module.set_return_attention(False)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 不使用异步执行
    ######################################

    ###############训练###############
    for i in range(model_parse.epoc):
        step = 0
        start_time = time.time()
        module.train()  # 可以不写 但最好写
        ###############加载数据集###############
        merged_train = DataLoader(dataset=train_data, batch_size=model_parse.batch_size, shuffle=True, drop_last=True)
        merged_test = DataLoader(dataset=test_data, batch_size=model_parse.batch_size, shuffle=True, drop_last=True)
        merged_validate = DataLoader(dataset=validate_data, batch_size=model_parse.batch_size, shuffle=True,
                                     drop_last=True)
        ######################################

        epoch_loss = 0.0
        print("__________________________________________epoch{}__________________________________________".format(i))
        print("__________________________________________epoch{}__________________________________________".format(i),
              file=logfile)

        if stop_flag == model_parse.patience:
            if lr_flag < 2:
                stop_flag = 0
                print("_________________________________________________________________________________________")
                print("_________________________________________________________________________________________",
                      file=logfile)
                print("已经超过{}轮模型没有得到改进，将修改学习率".format(model_parse.patience))
                print("已经超过{}轮模型没有得到改进，将修改学习率".format(model_parse.patience), file=logfile)
                print("总共运行了{}轮".format(i))
                print("总共运行了{}轮".format(i), file=logfile)
                print("最小损失：{}".format(min_loss))
                print("最小损失：{}".format(min_loss), file=logfile)
                print("最优模型:")
                print("最优模型:", file=logfile)
                print(best_dict)
                print(best_dict, file=logfile)
                print("验证最优模型:")
                print("验证最优模型:", file=logfile)
                print(best_valid)
                print(best_valid, file=logfile)
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
                lr_flag += 1
            else:
                print("_________________________________________________________________________________________")
                print("_________________________________________________________________________________________",
                      file=logfile)
                print("已经超过{}轮模型没有得到改进，将停止运行".format(model_parse.patience))
                print("已经超过{}轮模型没有得到改进，将停止运行".format(model_parse.patience), file=logfile)
                print("总共运行了{}轮".format(i))
                print("总共运行了{}轮".format(i), file=logfile)
                print("最小损失：{}".format(min_loss))
                print("最小损失：{}".format(min_loss), file=logfile)
                print("最优模型:")
                print("最优模型:", file=logfile)
                print(best_dict)
                print(best_dict, file=logfile)
                print("验证最优模型:")
                print("验证最优模型:", file=logfile)
                print(best_valid)
                print(best_valid, file=logfile)

                # 获取注意力权重矩阵并保存
                print("________________________________________获取注意力权重矩阵并保存_________________________________________________")
                getAttrJson(module, model_parse, path_parse, "/home/ysusmart/xhz/HTN-master/attrWidget0_1.json",
                            rate=0.001)
                getAttrJson(module,model_parse,path_parse,"/home/ysusmart/xhz/HTN-master/attrWidget.json",rate = 0.05)
                exit(0)
                logfile.close()

        train_labels = []  # 真实标签
        train_pred = []  # 预测标签
        train_preds = []  # 模型的打分结果中类别1的概率，是一个n行 ，1列的数组

        test_labels = []  # 真实标签
        test_pred = []  # 预测标签
        test_preds = []  # 模型的打分结果中类别1的概率，是一个n行 ，1列的数组

        valid_labels = []  # 真实标签
        valid_pred = []  # 预测标签
        valid_preds = []  # 模型的打分结果中类别1的概率，是一个n行 ，1列的数组

        for batch in merged_train:
            step += 1
            data, tag, ret = batch
            if torch.cuda.is_available():
                data = data.cuda()
                # tag1 = tag1.cuda()
                tag = tag.cuda()
                ret = ret.cuda()
                # tag3 = tag3.cuda()

            output = module(data, ret)
            # output,line_attention_weights,hunk_attention_weights = module(data, ret,True)

            targets = tag.float()
            loss = loss_fn(output, targets)
            optimizer.zero_grad()  # 优化过程中首先要使用优化器进行梯度清零
            loss.backward()  # 调用得到的损失，利用反向传播，得到每一个参数节点的梯度
            optimizer.step()  # 更新梯度

            epoch_loss = epoch_loss + loss.item()  # item是转为数字

            train_labels.extend(torch.Tensor.cpu(tag[:, 1]).detach().numpy())
            train_preds.extend(torch.Tensor.cpu(output[:, 1]).detach().numpy())
            predicted = torch.argmax(output, 1)
            train_pred.extend(torch.Tensor.cpu(predicted).detach().numpy())

            if step % 500 == 0:
                print("step:{}   loss:{}".format(step, loss.item()))
                print("step:{}   loss:{}".format(step, loss.item()), file=logfile)
        print("_________________训练信息_________________")
        print("_________________训练信息_________________", file=logfile)
        train_dict = {
            'roc': metrics.roc_auc_score(train_labels, train_preds),
            'acc': metrics.accuracy_score(train_labels, train_pred),
            'precision': metrics.precision_score(train_labels, train_pred, zero_division=0.0),
            'recall': metrics.recall_score(train_labels, train_pred, zero_division=0.0),
            'f1': metrics.f1_score(train_labels, train_pred, zero_division=0.0),
            'score': 0,
            'epoc': i
        }
        train_dict = getscore(train_dict)
        print(train_dict)
        print(train_dict, file=logfile)
        epoch_loss = epoch_loss / len(merged_train)
        print("训练总损失：{}".format(epoch_loss))
        print("训练总损失：{}".format(epoch_loss), file=logfile)

        # 测试模型
        module.eval()
        with torch.no_grad():  # 不计算梯度，用于推理和验证
            # torch.cuda.empty_cache()
            sum_loss = 0.0

            for batch in merged_test:
                # torch.cuda.empty_cache()
                data, tag, ret = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    tag = tag.cuda()
                    ret = ret.cuda()
                output = module(data, ret)
                # 保存标签
                test_labels.extend(torch.Tensor.cpu(tag[:, 1]).detach().numpy())
                test_preds.extend(torch.Tensor.cpu(output[:, 1]).detach().numpy())
                predicted = torch.argmax(output, 1)
                test_pred.extend(torch.Tensor.cpu(predicted).detach().numpy())
                loss = loss_fn(output, tag.float())
                sum_loss = sum_loss + loss.data
            # 统计结果
            print("_________________测试信息_________________")
            print("_________________测试信息_________________", file=logfile)
            test_dict = {
                'roc': metrics.roc_auc_score(test_labels, test_preds),
                'acc': metrics.accuracy_score(test_labels, test_pred),
                'precision': metrics.precision_score(test_labels, test_pred, zero_division=0.0),
                'recall': metrics.recall_score(test_labels, test_pred, zero_division=0.0),
                'f1': metrics.f1_score(test_labels, test_pred, zero_division=0.0),
                'score': 0,
                'epoc': i
            }
            test_dict = getscore(test_dict)
            print(test_dict)
            print(test_dict, file=logfile)
            sum_loss = sum_loss / len(merged_test)
            print('验证总损失:', sum_loss.item())
            print('验证总损失:', sum_loss.item(), file=logfile)
            if is_best(test_dict, best_dict):
                best_dict = test_dict
                # 在这里加一段保存模型
                stop_flag = 0
            else:
                stop_flag += 1
            min_loss = min(sum_loss, min_loss)
            print("_________________测试最优模型_________________")
            print("_________________测试最优模型_________________", file=logfile)

            print("最小损失：{}".format(min_loss))
            print("最小损失：{}".format(min_loss), file=logfile)
            print(best_dict)
            print(best_dict, file=logfile)

        module.eval()
        with torch.no_grad():  # 不计算梯度，用于推理
            sum_loss = 0.0
            for batch in merged_validate:
                data, tag, ret = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    tag = tag.cuda()
                    ret = ret.cuda()

                    # targets = tag.float()
                output = module(data, ret)

                # 保存标签
                valid_labels.extend(torch.Tensor.cpu(tag[:, 1]).detach().numpy())
                valid_preds.extend(torch.Tensor.cpu(output[:, 1]).detach().numpy())
                predicted = torch.argmax(output, 1)
                valid_pred.extend(torch.Tensor.cpu(predicted).detach().numpy())
                loss = loss_fn(output, tag.float())
                sum_loss = sum_loss + loss.data
            # 统计结果
            print("_________________验证信息_________________")
            print("_________________验证信息_________________", file=logfile)

            valid_dict = {
                'roc': metrics.roc_auc_score(valid_labels, valid_preds),
                'acc': metrics.accuracy_score(valid_labels, valid_pred),
                'precision': metrics.precision_score(valid_labels, valid_pred, zero_division=0.0),
                'recall': metrics.recall_score(valid_labels, valid_pred, zero_division=0.0),
                'f1': metrics.f1_score(valid_labels, valid_pred, zero_division=0.0),
                'score': 0,
                'epoc': i
            }
            valid_dict = getscore(valid_dict)
            print(valid_dict)
            print(valid_dict, file=logfile)

            sum_loss = sum_loss / len(merged_test)
            print('验证总损失:', sum_loss.item())
            print('验证总损失:', sum_loss.item(), file=logfile)
            if is_best(valid_dict, best_valid):
                best_valid = valid_dict
                # 在这里加一段保存模型
                stop_flag = 0
            else:
                stop_flag += 1
            min_loss = min(sum_loss, min_loss)
            print("_________________验证最优模型_________________")
            print("_________________验证最优模型_________________", file=logfile)

            print("最小损失：{}".format(min_loss))
            print("最小损失：{}".format(min_loss), file=logfile)
            print(best_valid)
            print(best_valid, file=logfile)

            end_time = time.time()
        print("本轮总运行时间{}".format(end_time - start_time))
        print("本轮总运行时间{}".format(end_time - start_time), file=logfile)
