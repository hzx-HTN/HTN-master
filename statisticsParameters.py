import torch
import torch.nn as nn
from model import MyModule
from parse import HTN_single_layer_parse


def a_static(model):
    # 统计总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 如果需要区分可训练参数和不可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    return [trainable_params, non_trainable_params]






if __name__ == '__main__':
    myModel_parse = HTN_single_layer_parse()
    myModel = MyModule(myModel_parse)
    a_static(myModel)


