import torch
import itertools
import unicodedata
import re
from config import *

# 0填充
def zeroPadding(l, fillvalue=0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# 生成mask
def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_TOKEN:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def sentence2indexes(voc, sentence):
    '''将句子转化成编码'''
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_TOKEN]


def inputTensor(sentence_batch, voc):
    '''
    生成输入张量
    :param sentence_batch: 语句列表
    :param voc: 单词表, Vocabulary对象
    :return: 输入张量(max_length, batch_size)，每个语句单词长度
    '''
    indexes_batch = [sentence2indexes(voc, sentence) for sentence in sentence_batch]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch) #转置并填充0-->shape = (max_length, batch_size)
    pad_tensor = torch.LongTensor(pad_list)
    return pad_tensor, lengths

def targetTensor(sentence_batch, voc):
    '''
    生成输出张量与mask
    :param sentence_batch: 语句列表
    :param voc: 单词表, Vocabulary对象
    :return: 输入张量(max_length, batch_size), mask, 最长单词数
    '''
    indexes_batch = [sentence2indexes(voc, sentence) for sentence in sentence_batch]
    max_target_lengths = max([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch)
    mask = binaryMatrix(pad_list)
    mask = torch.ByteTensor(mask)
    pad_tensor = torch.LongTensor(pad_list)
    return pad_tensor, mask, max_target_lengths

def batch2TrainData(voc, pair_batch):
    '''
    将问答对组转化成可训练的张量
    :param voc: 单词表
    :param pair_batch: 一组问答对
    :return: 输入张量，输入张量长度， 输出张量， mask, 输出张量最长长度
    '''
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True) # 单词数从大到小排序
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    input, lengths = inputTensor(input_batch, voc)
    output, mask, max_target_length = targetTensor(output_batch, voc)
    return input, lengths, output, mask, max_target_length


def unicodeToAscii(s):
    '''用于将unicode字符串转换成ascii'''
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    '''用于处理句子'''
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1 ", s) # 符号前加空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) #只保留a~z, A~Z, .!?, 其他替换为空格
    s = re.sub(r'\s+', r" ", s).strip() # 多个空格替换为一个
    return s

def normalizeInputChinese(s):
    return ' '.join(list(s)).strip()

def filterPair(p):
    return len(p[0].split()) < MAX_WORDS and len(p[1].split()) < MAX_WORDS

def fileterPairs(pairs):
    '''过滤过长语句'''
    return [pair for pair in pairs if filterPair(pair)]
