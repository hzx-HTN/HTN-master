import os
import time

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn import metrics

from model import GRUClassifier
from parse import GRU_parse
from CodeXGLUEutils import MyDataset, file_parse_noNorm, is_best, getscore
from CWEutils import CWEfile_parse,CWEBalancefile_parse,MyCWEDataset

if __name__ == '__main__':
    path_parse = CWEBalancefile_parse()
    model_parse = GRU_parse()
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
    module = GRUClassifier(model_parse)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.ASGD(module.parameters(), lr=model_parse.learn_rate)
    optimizer = torch.optim.Adam(module.parameters(), lr=model_parse.learn_rate)
    ####################################

    # ###############加载数据集###############
    test_data = MyCWEDataset(path_parse.split_data_path + "//test.json", path_parse.dict_path, model_parse, dim=1,
                          need_clean=False)
    train_data = MyCWEDataset(path_parse.split_data_path + "//train.json", path_parse.dict_path, model_parse, dim=1,
                           need_clean=False)
    validate_data = MyCWEDataset(path_parse.split_data_path + "//valid.json", path_parse.dict_path, model_parse, dim=1,
                              need_clean=False)
    # ######################################

    ###############cuda化###############
    module = module.cuda()
    loss_fn = loss_fn.cuda()
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
        if stop_flag == model_parse.patience:
            if lr_flag < 2:
                stop_flag = 0
                print("_________________________________________________________________________________________")
                print("已经超过{}轮模型没有得到改进，将修改学习率".format(model_parse.patience))
                print("总共运行了{}轮".format(i))
                print("最小损失：{}".format(min_loss))
                print("最优模型:")
                print(best_dict)
                print("验证最优模型:")
                print(best_valid)
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
                lr_flag += 1
            else:
                print("_________________________________________________________________________________________")
                print("已经超过{}轮模型没有得到改进，将停止运行".format(model_parse.patience))
                print("总共运行了{}轮".format(i))
                print("最小损失：{}".format(min_loss))
                print("最小损失：{}".format(min_loss))
                print("最优模型:")
                print(best_dict)
                print("验证最优模型:")
                print(best_valid)
                exit(0)

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
            data, tag = batch
            # print("获取数据成功{}".format(step))
            if torch.cuda.is_available():
                data = data.cuda()
                # tag1 = tag1.cuda()
                tag = tag.cuda()
                # tag3 = tag3.cuda()

            output = module(data)
            # print("成功进行一次计算")
            targets = tag.float()
            loss = loss_fn(output, targets)
            # print("损失计算成功")
            optimizer.zero_grad()  # 优化过程中首先要使用优化器进行梯度清零
            # print("梯度清零成功")
            loss.backward()  # 调用得到的损失，利用反向传播，得到每一个参数节点的梯度
            # print("反向传播成功")
            optimizer.step()  # 更新梯度
            # print("更新梯度成功")

            epoch_loss = epoch_loss + loss.item()  # item是转为数字

            train_labels.extend(torch.Tensor.cpu(tag[:, 1]).detach().numpy())
            train_preds.extend(torch.Tensor.cpu(output[:, 1]).detach().numpy())
            predicted = torch.argmax(output, 1)
            train_pred.extend(torch.Tensor.cpu(predicted).detach().numpy())

            if step % 2000 == 0:
                print("step:{}   loss:{}".format(step, loss.item()))
        print("_________________训练信息_________________")
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
        epoch_loss = epoch_loss / len(merged_train)
        print("训练总损失：{}".format(epoch_loss))

        # 测试模型
        module.eval()
        with torch.no_grad():  # 不计算梯度，用于推理和验证
            # torch.cuda.empty_cache()
            sum_loss = 0.0
            for batch in merged_test:
                # torch.cuda.empty_cache()
                data, tag = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    tag = tag.cuda()

                    # targets = tag.float()
                output = module(data)
                # 保存标签
                test_labels.extend(torch.Tensor.cpu(tag[:, 1]).detach().numpy())
                test_preds.extend(torch.Tensor.cpu(output[:, 1]).detach().numpy())
                predicted = torch.argmax(output, 1)
                test_pred.extend(torch.Tensor.cpu(predicted).detach().numpy())
                loss = loss_fn(output, tag.float())
                sum_loss = sum_loss + loss.data
            # 统计结果
            print("_________________测试信息_________________")
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
            sum_loss = sum_loss / len(merged_test)
            print('验证总损失:', sum_loss.item())
            if is_best(test_dict, best_dict):
                best_dict = test_dict
                # 在这里加一段保存模型
                stop_flag = 0
            else:
                stop_flag += 1
            min_loss = min(sum_loss, min_loss)
            print("_________________测试最优模型_________________")
            print("最小损失：{}".format(min_loss))
            print(best_dict)
        module.eval()
        with torch.no_grad():  # 不计算梯度，用于推理
            # torch.cuda.empty_cache()
            sum_loss = 0.0
            for batch in merged_validate:
                # torch.cuda.empty_cache()
                data, tag = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    tag = tag.cuda()

                    # targets = tag.float()
                output = module(data)
                # 保存标签
                valid_labels.extend(torch.Tensor.cpu(tag[:, 1]).detach().numpy())
                valid_preds.extend(torch.Tensor.cpu(output[:, 1]).detach().numpy())
                predicted = torch.argmax(output, 1)
                valid_pred.extend(torch.Tensor.cpu(predicted).detach().numpy())
                loss = loss_fn(output, tag.float())
                sum_loss = sum_loss + loss.data
            # 统计结果
            print("_________________验证信息_________________")
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
            sum_loss = sum_loss / len(merged_test)
            print('验证总损失:', sum_loss.item())
            if is_best(valid_dict, best_valid):
                best_valid = valid_dict
                # 在这里加一段保存模型
                stop_flag = 0
            else:
                stop_flag += 1
            min_loss = min(sum_loss, min_loss)
            print("_________________验证最优模型_________________")
            print("最小损失：{}".format(min_loss))
            print(best_valid)
            end_time = time.time()
        print("本轮总运行时间{}".format(end_time - start_time))
