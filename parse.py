import argparse


# dict1 9701
# dict2 143456
# cwedict 1343442
# cwebalancedict 763653
# 483813
def tokenCNN_parse():
    parser = argparse.ArgumentParser(description='TokenCNN model parse.')
    # 模型参数
    parser.add_argument('-code_len', default=512,
                        help='The maximum length of a piece of code.', type=int)
    parser.add_argument('-embed_size', default=13,
                        help='The maximum length of a piece of code.', type=int)
    parser.add_argument('-kernel_length', type=int, default=9, help='卷积核长度')
    parser.add_argument('-number_of_kernel', type=int, default=512, help='卷积核数量')
    parser.add_argument('-dict_len', default=483813,
                        help='dict length', type=int)

    # 训练参数
    parser.add_argument('-learn_rate', default=1e-3,
                        help='learning rate.', type=float)
    parser.add_argument('-batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-epoc', type=int, default=10000, help='最大训练轮数')
    parser.add_argument('-patience', type=int, default=20, help='早停耐心')

    args = parser.parse_args()
    return args


def GRU_parse():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('-embed_size', type=int, default=64, help='嵌入的维度')
    parser.add_argument('-code_len', type=int, default=512, help='代码长度')
    parser.add_argument('-dict_len', type=int, default=483813, help='字典长度')
    parser.add_argument('-hidden_size', type=int, default=256, help='隐藏层的大小，即GRU中神经元的数量')
    parser.add_argument('-num_layers', type=int, default=1, help='GRU层的数量')
    parser.add_argument('-drop_rate', type=float, default=0.2, help='GRU层的数量')

    # 训练参数
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-epoc', type=int, default=1000, help='规定一个代码段最多有多少行')
    parser.add_argument('-learn_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('-patience', type=int, default=20, help='早停耐心')

    args = parser.parse_args()
    return args


def LSTM_parse():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('-embed_size', type=int, default=50, help='嵌入的维度')
    parser.add_argument('-code_len', type=int, default=1024, help='代码长度')
    parser.add_argument('-dict_len', type=int, default=483813, help='字典长度')
    parser.add_argument('-hidden_size', type=int, default=64, help='隐藏层的大小，即GRU中神经元的数量')
    parser.add_argument('-num_layers', type=int, default=1, help='GRU层的数量')

    # 训练参数
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-epoc', type=int, default=1000, help='规定一个代码段最多有多少行')
    parser.add_argument('-learn_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('-patience', type=int, default=20, help='早停耐心')

    args = parser.parse_args()
    return args


def HANDL_parse():
    parser = argparse.ArgumentParser()

    # 模型参数
    # 数据集没啥用
    parser.add_argument('-dataset', type=str, default='activemq', help='software project name (lowercase)')

    parser.add_argument('-embed_size', type=int, default=50, help='词嵌入的维度')
    parser.add_argument('-hidden_size', type=int, default=64, help='')
    # parser.add_argument('-word_gru_num_layers', type=int, default=1, help='词GRU层数')
    parser.add_argument('-dropout', type=float, default=0.2, help='dropout rate 丢弃率')
    parser.add_argument('-dict_len', type=int, default=483813, help='字典长度')
    parser.add_argument('-max_slice_num', type=int, default=64, help='最大切片数量')
    parser.add_argument('-class_num', type=int, default=2)

    parser.add_argument('-code_len', type=int, default=64, help='代码长度')
    parser.add_argument('-line_len', type=int, default=10)

    # 训练参数
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-epoc', type=int, default=1000, help='规定一个代码段最多有多少行')
    parser.add_argument('-learn_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('-patience', type=int, default=20, help='早停耐心')

    args = parser.parse_args()
    return args



def HANCC2Vec_parse():
    parser = argparse.ArgumentParser()

    # 模型参数
    # 数据集没啥用
    parser.add_argument('-dataset', type=str, default='activemq', help='software project name (lowercase)')

    parser.add_argument('-embed_size', type=int, default=64, help='词嵌入的维度')
    parser.add_argument('-hidden_size', type=int, default=32, help='')
    # parser.add_argument('-word_gru_num_layers', type=int, default=1, help='词GRU层数')
    parser.add_argument('-dropout', type=float, default=0.2, help='dropout rate 丢弃率')
    parser.add_argument('-dict_len', type=int, default=483813, help='字典长度')
    parser.add_argument('-max_slice_num', type=int, default=64, help='最大切片数量')
    parser.add_argument('-class_num', type=int, default=2)

    parser.add_argument('-code_len', type=int, default=64, help='代码长度')
    parser.add_argument('-line_len', type=int, default=10)

    # 训练参数
    parser.add_argument('-batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-epoc', type=int, default=1000, help='训练轮数')
    parser.add_argument('-learn_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('-patience', type=int, default=20, help='早停耐心')

    args = parser.parse_args()
    return args

def HTNCWE_parse():
    """ CWE best parse"""
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('-embed_size', type=int, default=512, help='嵌入的维度')
    parser.add_argument('-code_len', type=int, default=64, help='代码段最大行数')
    parser.add_argument('-line_len', type=int, default=32, help='代码行最大长度')
    parser.add_argument('-dict_len', type=int, default=483813, help='字典长度')
    parser.add_argument('-retraction_len', type=int, default=60, help='缩进最大值')
    parser.add_argument('-line_layer_num', type=int, default=8, help='Line Encoder 中 Transformer层的数量')
    parser.add_argument('-hunk_layer_num', type=int, default=8, help='Hunk Encoder 中 Transformer层的数量')
    parser.add_argument('-head_num', type=int, default=8, help='注意力头的数量')
    parser.add_argument('-pos_drop', type=float, default=0.1, help='位置编码中的丢弃率')
    # parser.add_argument('-middle_drop', type=float, default=0.1, help='位置编码中的丢弃率')
    parser.add_argument('-final_drop', type=float, default=0.2, help='最后全连接之前的丢弃')

    # 训练参数
    parser.add_argument('-batch_size', type=int, default=24, help='batch size')
    parser.add_argument('-epoc', type=int, default=1000, help='最大训练轮数')
    parser.add_argument('-patience', type=int, default=10, help='早停耐心')
    parser.add_argument('-learn_rate', type=float, default=1e-6, help='学习率')

    args = parser.parse_args()
    return args

def HTN_parse():
    """ CWE 消融实验参数"""
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('-embed_size', type=int, default=512, help='嵌入的维度')
    parser.add_argument('-code_len', type=int, default=31, help='代码段最大行数')
    parser.add_argument('-line_len', type=int, default=32, help='代码行最大长度')
    parser.add_argument('-dict_len', type=int, default=483813, help='字典长度')
    parser.add_argument('-retraction_len', type=int, default=60, help='缩进最大值')
    parser.add_argument('-line_layer_num', type=int, default=8, help='Line Encoder 中 Transformer层的数量')
    parser.add_argument('-hunk_layer_num', type=int, default=8, help='Hunk Encoder 中 Transformer层的数量')
    parser.add_argument('-head_num', type=int, default=8, help='注意力头的数量')
    parser.add_argument('-pos_drop', type=float, default=0.1, help='位置编码中的丢弃率')
    # parser.add_argument('-middle_drop', type=float, default=0.1, help='位置编码中的丢弃率')
    parser.add_argument('-final_drop', type=float, default=0.2, help='最后全连接之前的丢弃')

    # 训练参数
    parser.add_argument('-batch_size', type=int, default=24, help='batch size')
    parser.add_argument('-epoc', type=int, default=1000, help='最大训练轮数')
    parser.add_argument('-patience', type=int, default=10, help='早停耐心')
    parser.add_argument('-learn_rate', type=float, default=1e-6, help='学习率')

    args = parser.parse_args()
    return args

def HTN_single_layer_parse():
    """ 单层模型参数"""
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('-embed_size', type=int, default=512, help='嵌入的维度')
    parser.add_argument('-code_len', type=int, default=64, help='代码段最大行数')
    parser.add_argument('-line_len', type=int, default=32, help='代码行最大长度')
    parser.add_argument('-dict_len', type=int, default=483813, help='字典长度')
    parser.add_argument('-retraction_len', type=int, default=60, help='缩进最大值')
    parser.add_argument('-line_layer_num', type=int, default=4, help='Line Encoder 中 Transformer层的数量')
    parser.add_argument('-hunk_layer_num', type=int, default=4, help='Hunk Encoder 中 Transformer层的数量')
    parser.add_argument('-head_num', type=int, default=4, help='注意力头的数量')
    parser.add_argument('-pos_drop', type=float, default=0.1, help='位置编码中的丢弃率')
    # parser.add_argument('-middle_drop', type=float, default=0.1, help='位置编码中的丢弃率')
    parser.add_argument('-final_drop', type=float, default=0.2, help='最后全连接之前的丢弃')

    # 训练参数
    parser.add_argument('-batch_size', type=int, default=24, help='batch size')
    parser.add_argument('-epoc', type=int, default=1000, help='最大训练轮数')
    parser.add_argument('-patience', type=int, default=10, help='早停耐心')
    parser.add_argument('-learn_rate', type=float, default=1e-6, help='学习率')

    args = parser.parse_args()
    return args

def HTNp_parse():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('-embed_size', type=int, default=128, help='嵌入的维度')
    parser.add_argument('-hidden_size', type=int, default=256, help='rnn隐藏维度')
    parser.add_argument('-rnn_num_layers', type=int, default=1, help='rnn层数')
    parser.add_argument('-code_len', type=int, default=64, help='代码段最大行数')
    parser.add_argument('-line_len', type=int, default=16, help='代码行最大长度')
    parser.add_argument('-dict_len', type=int, default=483813, help='字典长度')
    parser.add_argument('-retraction_len', type=int, default=26, help='缩进最大值')
    parser.add_argument('-line_layer_num', type=int, default=6, help='Line Encoder 中 Transformer层的数量')
    parser.add_argument('-hunk_layer_num', type=int, default=6, help='Hunk Encoder 中 Transformer层的数量')
    parser.add_argument('-head_num', type=int, default=8, help='注意力头的数量')
    parser.add_argument('-pos_drop', type=float, default=0.01, help='位置编码中的丢弃率')
    # parser.add_argument('-middle_drop', type=float, default=0.1, help='位置编码中的丢弃率')
    parser.add_argument('-final_drop', type=float, default=0.2, help='最后全连接之前的丢弃')

    # 训练参数
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-epoc', type=int, default=1000, help='最大训练轮数')
    parser.add_argument('-patience', type=int, default=200, help='早停耐心')
    parser.add_argument('-learn_rate', type=float, default=1e-6, help='学习率')

    args = parser.parse_args()
    return args

# 使用示例
def BERT_parse():
    """BERT 单层模型参数"""
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('-embed_size', type=int, default=512, help='嵌入的维度')
    parser.add_argument('-code_len', type=int, default=64*32, help='代码行最大长度')
    parser.add_argument('-dict_len', type=int, default=483813, help='字典长度')
    parser.add_argument('-layer_num', type=int, default=8, help='Line Encoder 中 Transformer层的数量')
    parser.add_argument('-head_num', type=int, default=4, help='注意力头的数量')
    parser.add_argument('-pos_drop', type=float, default=0.1, help='位置编码中的丢弃率')
    parser.add_argument('-middle_drop', type=float, default=0.1, help='位置编码中的丢弃率')
    parser.add_argument('-final_drop', type=float, default=0.2, help='最后全连接之前的丢弃')

    # 训练参数
    parser.add_argument('-batch_size', type=int, default=24, help='batch size')
    parser.add_argument('-epoc', type=int, default=1000, help='最大训练轮数')
    parser.add_argument('-patience', type=int, default=10, help='早停耐心')
    parser.add_argument('-learn_rate', type=float, default=1e-6, help='学习率')

    args = parser.parse_args()
    return args
