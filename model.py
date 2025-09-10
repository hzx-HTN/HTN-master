import math
import os
import pickle
import torch
from random import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from CodeXGLUEutils import file_parse
import torch.nn.functional as F
from torch.autograd import Variable
from parse import tokenCNN_parse, GRU_parse


# --------------------Model TokenCNN--------------------
class TokenCNN(nn.Module):
    def __init__(self, parser):
        super(TokenCNN, self).__init__()
        # self.flatten = nn.Flatten(1)
        self.src_emb = nn.Embedding(parser.dict_len,
                                    parser.embed_size)
        self.conv = nn.Conv2d(in_channels=1, out_channels=parser.number_of_kernel,
                              kernel_size=(parser.kernel_length, parser.embed_size))
        self.pool = nn.MaxPool2d(kernel_size=(504, 1))  # 512
        # self.pool = nn.MaxPool2d(kernel_size=(1016, 1))  # 1024
        self.linear = nn.Sequential(
            nn.Dropout(p=0.3),
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


# --------------------Model CodeGRU--------------------
class GRUClassifier(nn.Module):
    def __init__(self, parser):
        super(GRUClassifier, self).__init__()
        self.src_emb = nn.Embedding(parser.dict_len,
                                    parser.embed_size)
        self.rnn = nn.GRU(parser.embed_size, parser.hidden_size, batch_first=True, bidirectional=True)
        self.h_0 = Variable(torch.zeros(2, parser.batch_size, parser.hidden_size).cuda())
        self.linear = nn.Sequential(
            nn.Dropout(p=parser.drop_rate),
            nn.Linear(parser.hidden_size * 2, parser.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(parser.hidden_size, 16),
            nn.Sigmoid(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

    def forward(self, code_tensor):
        out = self.src_emb(code_tensor)
        __, hidden_state = self.rnn(out, self.h_0)
        # hidden_state = hidden_state.squeeze(0)
        hidden_state = hidden_state.permute(1, 2, 0)
        hidden_state = torch.flatten(hidden_state, start_dim=1)
        out = self.linear(hidden_state)
        return out


# --------------------Model BiLSTM--------------------
class LSTMClassifier(nn.Module):
    def __init__(self, args):
        super(LSTMClassifier, self).__init__()

        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.vocab_size = args.dict_len
        self.embedding_length = args.embed_size

        # Initializing the look-up table.
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional=True, batch_first=True)
        self.h_0 = Variable(
            torch.zeros(2, self.batch_size, self.hidden_size).cuda())  # Initialize hidden state of the LSTM
        self.c_0 = Variable(
            torch.zeros(2, self.batch_size, self.hidden_size).cuda())  # Initialize cell state of the LSTM
        # dropout layer
        self.dropout = nn.Dropout(0.2)
        # linear and sigmoid layer
        self.mlp = nn.Linear(self.hidden_size * 2, 2)
        self.sig = nn.Sigmoid()

    def forward(self, input_tensor):
        # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = self.word_embeddings(input_tensor.type(torch.LongTensor).cuda())

        # input.size() = (num_sequences, batch_size, embedding_length)

        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(input, (self.h_0, self.c_0))
        output = final_hidden_state.permute(1, 2, 0)
        output = torch.flatten(output, start_dim=1)
        output = self.mlp(output)  # the last hidden state is output of lstm model

        sig_out = self.sig(output)

        return sig_out


# --------------------Model HierarchicalAttentionNetwork--------------------
# Make the the multiple attention with word vectors.
# 融合注意力权重
def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(1)):
        h_i = rnn_outputs[:, i, :]
        a_i = att_weights[:, i, :]
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(1)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 1)
    out = torch.sum(attn_vectors, 1)
    out = out.unsqueeze(1)
    return out


# WordRNN单词生成嵌入 SentRNN同理
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, batch_size, hidden_size):
        super(WordRNN, self).__init__()
        # 设置参数
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # Word Encoder
        # 嵌入 + GRU编码
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.wordRNN = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        # Word Attention
        # 死的注意力矩阵  + 全连接
        self.wordattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        # 嵌入构建
        emb_out = self.embed(inp)
        # 卷积
        out_state, hid_state = self.wordRNN(emb_out, hid_state)
        # 注意力
        word_annotation = self.wordattn(out_state)
        # 权重组合
        attn = F.softmax(self.attn_combine(word_annotation), dim=0)
        # 按权重相加
        sent = attention_mul(out_state, attn)
        return sent, hid_state


class SentRNN(nn.Module):
    def __init__(self, sent_size, hidden_size):
        super(SentRNN, self).__init__()
        # Sentence Encoder
        self.sent_size = sent_size
        self.sentRNN = nn.GRU(sent_size, hidden_size, batch_first=True, bidirectional=True)

        # Sentence Attention
        self.sentattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        out_state, hid_state = self.sentRNN(inp, hid_state)

        sent_annotation = self.sentattn(out_state)
        attn = F.softmax(self.attn_combine(sent_annotation), dim=0)

        sent = attention_mul(out_state, attn)
        return sent, hid_state


class HAN(nn.Module):
    def __init__(self, args):
        super(HAN, self).__init__()
        self.vocab_size = args.dict_len
        self.batch_size = args.batch_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.cls = args.class_num
        self.hid_state_sent = self.init_hidden_sent().cuda()
        self.hid_state_word = self.init_hidden_word().cuda()
        self.dropout = nn.Dropout(args.dropout)  # drop out

        # Word Encoder
        self.wordRNN = WordRNN(self.vocab_size, self.embed_size, self.batch_size, self.hidden_size)
        # Sentence Encoder
        self.sentRNN = SentRNN(self.hidden_size * 2, self.hidden_size)

        # Hidden layers before putting to the output layer

        self.fc1 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.fc2 = nn.Linear(2 * self.hidden_size, self.cls)
        self.sigmoid = nn.Sigmoid()

    def forward_code(self, x, hid_state):
        hid_state_sent, hid_state_word = hid_state
        n_batch, n_line = x.shape[0], x.shape[1]
        # i: hunk; j: line; k: batch
        sents = None
        for j in range(n_line):
            words = x[:, j, :]
            # words = np.array(words)
            # sent, state_word = self.wordRNN(torch.cuda.LongTensor(words).view(-1, self.batch_size), hid_state_word)

            sent, hid_state_word = self.wordRNN(words, hid_state_word)
            if sents is None:
                sents = sent
            else:
                sents = torch.cat((sents, sent), 1)
        # batch_size * 16
        hunk, hid_state_sent = self.sentRNN(sents, hid_state_sent)
        sents = torch.mean(sents, dim=0)  # hunk features
        return sents, hunk

    def forward(self, code):
        hid_state = (self.hid_state_sent, self.hid_state_word)
        x_mean_sent, x_hunk = self.forward_code(x=code, hid_state=hid_state)

        # 重构格式
        x_code = x_hunk.view(self.batch_size, self.hidden_size * 2)

        out = self.fc1(self.dropout(x_code))
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out

    def init_hidden_sent(self):
        return Variable(torch.zeros(2, self.batch_size, self.hidden_size))  # .cuda()

    def init_hidden_word(self):
        return Variable(torch.zeros(2, self.batch_size, self.hidden_size))  # .cuda()


# --------------------My Model--------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):  # 输入：嵌入长度，丢弃率，代码行最大长度
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LineEncoder(nn.Module):
    def __init__(self, parser):
        super(LineEncoder, self).__init__()
        self.parser = parser
        self.src_emb = nn.Embedding(parser.dict_len, parser.embed_size)
        self.pos_emb = PositionalEncoding(d_model=parser.embed_size, dropout=parser.pos_drop,
                                          max_len=parser.line_len)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=parser.embed_size, nhead=parser.head_num, batch_first=True) for _ in
             range(parser.line_layer_num)])
        self.mlp = nn.Sequential(
            nn.Linear(in_features=parser.embed_size, out_features=parser.embed_size * 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=parser.embed_size * 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=parser.embed_size, bias=True),
            nn.LeakyReLU()
        )

        self.relu = nn.LeakyReLU()

    def forward(self, inputs):  # inputs:line_number * line_len（128*128）
        outputs = []
        for i in range(self.parser.code_len):
            input = inputs[:, i, :]
            output = self.src_emb(input)  # output: vocab_num*d_module
            output = self.pos_emb(output.transpose(0, 1)).transpose(0, 1)  # output: vocab_num*d_module

            for layer in self.layers:
                output = layer(output)  # , src_key_padding_mask=attn_mask)  # output: vocab_num*d_module
                output = self.relu(output)
            output = output.sum(dim=1)
            # output = self.drop(output)  # (batch_size, d_model)
            outputs.append(output)
        # outputs = torch.stack(outputs, dim=1)
        outputs = self.mlp(torch.stack(outputs, dim=1))
        return outputs  # (batch_size,line_number,d_model)


class HunkEncoder(nn.Module):
    def __init__(self, parser):
        super(HunkEncoder, self).__init__()
        self.parser = parser
        self.pos_emb1 = PositionalEncoding(d_model=parser.embed_size, dropout=parser.pos_drop,
                                           max_len=parser.code_len)
        self.pos_emb2 = self.pos_emb2 = nn.Embedding(parser.retraction_len, parser.embed_size)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=parser.embed_size, nhead=parser.head_num, batch_first=True) for _ in
             range(parser.hunk_layer_num)])
        self.drop = nn.Dropout(p=parser.final_drop)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=parser.embed_size, out_features=parser.embed_size * 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=parser.embed_size * 2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=16, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=2, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU()

    def forward(self, input, retraction):
        output = self.pos_emb1(input.transpose(0, 1)).transpose(0, 1)  # 正余弦编码
        ret_emb = self.pos_emb2(retraction)  # (batch_size, line_number, d_model)
        output = output + ret_emb

        for layer in self.layers:
            output = layer(output)  # , src_key_padding_mask=attn_mask)
            output = self.relu(output)  # (batch_size, line_number, d_model)
        output = output.sum(dim=1)  # (batch_size, d_model)
        output = self.drop(output)  # (batch_size, d_model)
        output = self.mlp(output)  # (batch_size, 2)
        return output


class MyModule(nn.Module):  # 二分类模式
    def __init__(self, parser):
        super(MyModule, self).__init__()
        self.line_enc = LineEncoder(parser)
        self.hunk_enc = HunkEncoder(parser)

    def forward(self, inputs, retraction):
        output = self.line_enc(inputs)
        output = self.hunk_enc(output, retraction)
        return output


# --------------------My Model No-indentation--------------------
class MyModule_no_i(nn.Module):  # 二分类模式
    def __init__(self, parser):
        super(MyModule_no_i, self).__init__()
        self.line_enc = LineEncoder(parser)
        self.hunk_enc = HunkEncoder_no_i(parser)

    def forward(self, inputs, retraction):
        output = self.line_enc(inputs)
        output = self.hunk_enc(output, retraction)
        return output


class HunkEncoder_no_i(nn.Module):
    def __init__(self, parser):
        super(HunkEncoder_no_i, self).__init__()
        self.parser = parser
        self.pos_emb1 = PositionalEncoding(d_model=parser.embed_size, dropout=parser.pos_drop,
                                           max_len=parser.code_len)
        # self.pos_emb2 = self.pos_emb2 = nn.Embedding(parser.retraction_len, parser.embed_size)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=parser.embed_size, nhead=parser.head_num, batch_first=True) for _ in
             range(parser.hunk_layer_num)])
        self.drop = nn.Dropout(p=parser.final_drop)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=parser.embed_size, out_features=parser.embed_size * 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=parser.embed_size * 2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=16, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=2, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU()

    def forward(self, input, retraction):
        output = self.pos_emb1(input.transpose(0, 1)).transpose(0, 1)  # 正余弦编码
        # ret_emb = self.pos_emb2(retraction)  # (batch_size, line_number, d_model)
        # output = output + ret_emb

        for layer in self.layers:
            output = layer(output)  # , src_key_padding_mask=attn_mask)
            output = self.relu(output)  # (batch_size, line_number, d_model)
        output = output.sum(dim=1)  # (batch_size, d_model)
        output = self.drop(output)  # (batch_size, d_model)
        output = self.mlp(output)  # (batch_size, 2)
        return output


# def __init__(self, d_model, dropout, max_len):  # 输入：嵌入长度，丢弃率，代码行最大长度
class TokenCodeBERT(nn.Module):
    def __init__(self, parser):
        super(TokenCodeBERT, self).__init__()
        self.parser = parser
        self.src_emb = nn.Embedding(parser.dict_len, parser.embed_size)
        self.pos_emb = PositionalEncoding(d_model=parser.embed_size, dropout=parser.pos_drop,
                                          max_len=parser.code_len)

        # Transformer Encoder层
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=parser.embed_size, nhead=parser.head_num, batch_first=True) for _ in
             range(parser.layer_num)])

        self.drop = nn.Dropout(p=parser.final_drop)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=parser.embed_size, out_features=parser.embed_size * 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=parser.embed_size * 2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=16, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=2, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU()

    def forward(self, code_tensor):
        output = self.src_emb(code_tensor)
        output = self.pos_emb(output.transpose(0, 1)).transpose(0, 1)  # 正余弦编码

        for layer in self.layers:
            output = layer(output)  # , src_key_padding_mask=attn_mask)
            output = self.relu(output)  # (batch_size, line_number, d_model)

        output = output.sum(dim=1)  # (batch_size, d_model)
        output = self.drop(output)  # (batch_size, d_model)
        output = self.mlp(output)  # (batch_size, 2)
        return output


class LineEncoderAttr(nn.Module):
    def __init__(self, parser):
        super(LineEncoderAttr, self).__init__()
        self.parser = parser
        self.src_emb = nn.Embedding(parser.dict_len, parser.embed_size)
        self.pos_emb = PositionalEncoding(d_model=parser.embed_size, dropout=parser.pos_drop,
                                          max_len=parser.line_len)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=parser.embed_size, nhead=parser.head_num, batch_first=True) for _ in
             range(parser.line_layer_num)])
        self.mlp = nn.Sequential(
            nn.Linear(in_features=parser.embed_size, out_features=parser.embed_size * 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=parser.embed_size * 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=parser.embed_size, bias=True),
            nn.LeakyReLU()
        )
        self.relu = nn.LeakyReLU()
        self.return_attention = False

    def set_return_attention(self, return_attention=True):
        """设置是否返回注意力权重"""
        self.return_attention = return_attention

    def forward(self, inputs, return_attention=False):
        outputs = []
        all_attention_weights = []  # 存储所有注意力权重

        for i in range(self.parser.code_len):
            input = inputs[:, i, :]
            output = self.src_emb(input)
            output = self.pos_emb(output.transpose(0, 1)).transpose(0, 1)

            line_attention_weights = []  # 当前行的注意力权重

            for layer in self.layers:
                if return_attention or self.return_attention:  # 只在推理时计算注意力
                    attn_output, attn_weights = layer.self_attn(
                        output, output, output,
                        need_weights=True,
                        average_attn_weights=True
                    )

                    # 完成TransformerEncoderLayer的其余操作
                    output = layer.norm1(output + layer.dropout1(attn_output))
                    ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
                    output = layer.norm2(output + layer.dropout2(ff_output))
                    # np.array(attn_weights)
                    line_attention_weights.append(np.array(torch.Tensor.cpu(attn_weights[0].detach())))
                else:
                    output = layer(output)


                output = self.relu(output)

            output = output.sum(dim=1)
            outputs.append(output)

            if return_attention and not self.training:
                all_attention_weights.append(line_attention_weights)

        outputs = self.mlp(torch.stack(outputs, dim=1))

        if return_attention and not self.training:
            return outputs, all_attention_weights
        return outputs


class HunkEncoderAttn(nn.Module):
    def __init__(self, parser):
        super(HunkEncoderAttn, self).__init__()
        self.parser = parser
        self.pos_emb1 = PositionalEncoding(d_model=parser.embed_size, dropout=parser.pos_drop,
                                           max_len=parser.code_len)
        self.pos_emb2 = nn.Embedding(parser.retraction_len, parser.embed_size)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=parser.embed_size, nhead=parser.head_num, batch_first=True) for _ in
             range(parser.hunk_layer_num)])
        self.drop = nn.Dropout(p=parser.final_drop)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=parser.embed_size, out_features=parser.embed_size * 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=parser.embed_size * 2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=parser.embed_size * 2, out_features=16, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=2, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU()
        self.return_attention = False

    def set_return_attention(self, return_attention=True):
        """设置是否返回注意力权重"""
        self.return_attention = return_attention

    def forward(self, input, retraction, return_attention=False):
        output = self.pos_emb1(input.transpose(0, 1)).transpose(0, 1)  # 正余弦编码
        ret_emb = self.pos_emb2(retraction)  # (batch_size, line_number, d_model)
        output = output + ret_emb

        attention_weights = []  # 存储每一层的注意力权重

        for layer_idx, layer in enumerate(self.layers):
            if return_attention or self.return_attention:
                # 手动执行TransformerEncoderLayer的操作以获取注意力权重

                attn_output, attn_weights = layer.self_attn(
                    output, output, output,
                    need_weights=True,
                    average_attn_weights=True
                )

                attention_weights.append(np.array(torch.Tensor.cpu(attn_weights[0].detach())))

                # 2. Add & Norm (第一个残差连接)
                output = layer.norm1(output + layer.dropout1(attn_output))

                # 3. Feed Forward Network
                ff_output = layer.linear2(
                    layer.dropout(
                        layer.activation(layer.linear1(output))
                    )
                )

                # 4. Add & Norm (第二个残差连接)
                output = layer.norm2(output + layer.dropout2(ff_output))

            else:
                # 训练时使用原始的layer调用，不计算注意力权重，保持训练效率
                output = layer(output)

            output = self.relu(output)  # (batch_size, line_number, d_model)

        output = output.sum(dim=1)  # (batch_size, d_model)
        output = self.drop(output)  # (batch_size, d_model)
        output = self.mlp(output)  # (batch_size, 2)

        if return_attention or self.return_attention:
            return output, attention_weights
        return output


class HTN_Attr(nn.Module):  # 生成注意力矩阵模式
    def __init__(self, parser):
        super(HTN_Attr, self).__init__()
        self.line_enc = LineEncoderAttr(parser)
        self.hunk_enc = HunkEncoderAttn(parser)
        self.return_attention = False

    def forward(self, inputs, retraction, return_attention=False):
        outputs = []
        line_attention_weights = []
        hunk_attention_weights = []
        if return_attention or self.return_attention:
            output, line_attention_weights =self.line_enc(inputs, return_attention or self.return_attention)
            output, hunk_attention_weights = self.hunk_enc(output, retraction, return_attention)
            return output, line_attention_weights, hunk_attention_weights
        else:
            output = self.line_enc(inputs, return_attention)
            output = self.hunk_enc(output, retraction, return_attention)
            return output

    def set_return_attention(self, return_attention=True):
        """设置是否返回注意力权重"""
        self.return_attention = return_attention
        self.line_enc.set_return_attention(return_attention)
        self.hunk_enc.set_return_attention(return_attention)

if __name__ == '__main__':
    print("1")
