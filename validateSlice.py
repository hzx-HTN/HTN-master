import os
import time

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn import metrics
from torch.autograd import Variable
from model import HAN
from parse import HAN_parse
from CodeXGLUEutils import MyDataset, file_parse_noNorm, is_best, getscore

if __name__ == '__main__':
    path_parse = file_parse_noNorm()
    model_parse = HAN_parse()

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
    module = HAN(model_parse)
    module.load_state_dict(torch.load("./best_model/HAN_best5.pth"))
    state_word = module.init_hidden_word().cuda()
    state_sent = module.init_hidden_sent().cuda()

    ####################################

    # ###############加载数据集###############
    test_data = MyDataset(path_parse.split_data_path + "//test.json", path_parse.dict_path, model_parse, dim=2,
                          need_clean=True)
    test_data_sliced = MyDataset(path_parse.split_data_path + "//test.json", path_parse.dict_path, model_parse, dim=2,
                                 need_clean=True, need_slice=True)
    validate_data = MyDataset(path_parse.split_data_path + "//valid.json", path_parse.dict_path, model_parse, dim=2,
                              need_clean=True)
    validate_data_sliced = MyDataset(path_parse.split_data_path + "//valid.json", path_parse.dict_path, model_parse,
                                     dim=2,
                                     need_clean=True, need_slice=True)
    # ######################################

    ###############cuda化###############
    module = module.cuda()
    ######################################

    ###############加载数据集###############
    merged_test = DataLoader(dataset=validate_data, batch_size=model_parse.batch_size, shuffle=True, drop_last=True)
    merged_test_slice = DataLoader(dataset=validate_data_sliced, batch_size=model_parse.batch_size, shuffle=True,
                                   drop_last=True)
    merged_validate = DataLoader(dataset=validate_data, batch_size=model_parse.batch_size, shuffle=True,
                                 drop_last=True)
    merged_validate_slice = DataLoader(dataset=validate_data_sliced, batch_size=model_parse.batch_size, shuffle=True,
                                       drop_last=True)
    ######################################

    # 测试模型
    module.eval()
    # with torch.no_grad():  # 不计算梯度，用于推理和验证
    test_labels = []  # 真实标签
    test_pred = []  # 预测标签
    test_preds = []  # 模型的打分结果中类别1的概率，是一个n行 ，1列的数组
    # torch.cuda.empty_cache()
    sum_loss = 0.0
    for batch in merged_test:
        # torch.cuda.empty_cache()
        data, tag, retraction = batch
        if torch.cuda.is_available():
            data = data.cuda()
            tag = tag.cuda()

        output = module.forward(data, state_sent, state_word)
        # 保存标签
        test_labels.extend(torch.Tensor.cpu(tag[:, 1]).detach().numpy())
        # test_preds.extend(torch.Tensor.cpu(output[:, 1]).detach().numpy())
        predicted = torch.argmax(output, 1)
        test_pred.extend(torch.Tensor.cpu(predicted).detach().numpy())
    test_dict = {
        # 'roc': metrics.roc_auc_score(test_labels, test_preds),
        'acc': metrics.accuracy_score(test_labels, test_pred),
        'precision': metrics.precision_score(test_labels, test_pred, zero_division=0.0),
        'recall': metrics.recall_score(test_labels, test_pred, zero_division=0.0),
        'f1': metrics.f1_score(test_labels, test_pred, zero_division=0.0),
        'score': 0,
        'epoc': 0
    }
    print(test_dict)
    for batch in merged_test_slice:
        data_batch, tag_batch, retraction_batch, slice_num = batch
        tag = tag_batch.cuda()
        test_labels.extend(torch.Tensor.cpu(tag[:, 1]).detach().numpy())
        predicted = []
        for j in range(model_parse.batch_size):
            data = data_batch[j].cuda()
            output = module.forward(data, state_sent, state_word)
            # torch.Tensor.cpu(torch.argmax(output, 1))
            result = torch.Tensor.cpu(torch.argmax(output, 1)).detach().numpy()
            result = result[0:slice_num]
            if 1 in result:
                predicted.append(1)
            else:
                predicted.append(0)
        test_pred.extend(np.array(predicted))
    # 统计结果
    print("_________________测试信息_________________")
    test_dict = {
        # 'roc': metrics.roc_auc_score(test_labels, test_preds),
        'acc': metrics.accuracy_score(test_labels, test_pred),
        'precision': metrics.precision_score(test_labels, test_pred, zero_division=0.0),
        'recall': metrics.recall_score(test_labels, test_pred, zero_division=0.0),
        'f1': metrics.f1_score(test_labels, test_pred, zero_division=0.0),
        'score': 0,
        'epoc': 0
    }
    print(test_dict)
