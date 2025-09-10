import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

from parse import generic_parse, tokenCNN_parse, GRU_parse


class OneDimDataset(Dataset):
    def __init__(self, data_path, dict_path, parser):  # 初始化
        # 这个parser是模型的参数而不是通用参数
        self.filelist = os.listdir(data_path)
        self.data_path = data_path
        with open(dict_path, "rb") as file:
            self.dicts = pickle.load(file)
        file.close()
        self.len = len(self.filelist)
        self.parser = parser

    def __getitem__(self, index):
        # data_lists, tags, retraction = RebuildHdf5Data(self.data_path) # 放在初始化里防止重复计算
        # self.len = len(self.tags)
        self.itempath = self.data_path + "//" + self.filelist[index]
        with open(self.itempath, "rb") as file:
            data = pickle.load(file)
        file.close()
        code = self.rebuild_data(data['code'])
        # , data['retraction'] 在二维中缩进也需要处理成相同长度
        return code, torch.Tensor(data['tag'])

    def __len__(self):
        return self.len

    def rebuild_data(self, code):
        data = np.zeros(self.parser.code_len, dtype=int)
        data_length = 0
        for line in code:
            if data_length >= self.parser.code_len:
                break
            for vocab in line:
                if data_length >= self.parser.code_len:
                    break
                if line is not None:
                    data[data_length] = self.dicts[vocab]
                    data_length += 1
        return data


class TokenCNN(nn.Module):
    def __init__(self, parser):
        super(TokenCNN, self).__init__()
        # self.flatten = nn.Flatten(1)
        self.src_emb = nn.Embedding(parser.dict_len,
                                    parser.embed_size)
        self.conv = nn.Conv2d(in_channels=1, out_channels=parser.number_of_kernel,
                              kernel_size=(parser.kernel_length, parser.embed_size))
        self.pool = nn.MaxPool2d(kernel_size=(504, 1))
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

    def forward(self, code_tensor):
        # out = self.flatten(code_tensor)
        out = self.src_emb(code_tensor)
        out = out.unsqueeze(1)
        out = self.conv(out)
        out = self.pool(out)
        out = out.squeeze(3)
        out = out.squeeze(2)
        out = self.linear(out)
        return out


class CodeGRU(nn.Module):
    def __init__(self, parser):
        super(CodeGRU, self).__init__()
        self.flatten = nn.Flatten(1)
        self.src_emb = nn.Embedding(parser.dict_len,
                                    parser.embed_size)
        self.rnn = nn.GRU(parser.embed_size, parser.hidden_size, parser.num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

    def forward(self, code_tensor):
        out = self.src_emb(code_tensor)
        __, hidden_state = self.rnn(out)
        # hidden_state = hidden_state.squeeze(0)
        hidden_state = hidden_state.permute(1, 0, 2)
        hidden_state = self.flatten(hidden_state)
        out = self.linear(hidden_state)
        return out


def test():
    # x.permute(1, 0, 2)
    path_parse = generic_parse()
    model_parse = tokenCNN_parse()
    test_data = OneDimDataset(path_parse.split_data_path + "//test", path_parse.dict_path, model_parse)
    merged_data = DataLoader(dataset=test_data, batch_size=model_parse.batch_size, shuffle=True, drop_last=True)
    model = TokenCNN(model_parse)
    for batch in merged_data:
        code, tag = batch
        out = model(code)
        # print(out)
        # batch.shape()


if __name__ == '__main__':
    test()
