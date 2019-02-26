import csv
import os
from utils import *

def save_cornell_formatted(data_file):
    lines_filepath = os.path.join(DATASET_PATH, 'cornell', 'movie_lines.txt')
    conv_filepath = os.path.join(DATASET_PATH,'cornell', 'movie_conversations.txt')

    # 处理lines文件
    # L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
    print("Processing movie_lines.txt...")
    line_fields = ['lineID', 'characterID', 'movieID', 'character', "text"]
    lines = {}
    with open(lines_filepath, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            lineObj = {}
            for i, field in enumerate(line_fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj

    # 处理conversation文件
    # u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
    print("Processing movie_conversations.txt...\n")
    conv_fields = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']
    conversations = []
    with open(conv_filepath, 'r', encoding='iso-8859-1') as f:
        for line in f:
            value = line.split(' +++$+++ ')
            convObj = {}
            for i, field in enumerate(conv_fields):
                convObj[field] = value[i]
            lineIDs = eval(convObj['utteranceIDs']) # 将字符串推导为列表
            convObj['lines'] = []
            for lineID in lineIDs:
                convObj['lines'].append(lines[lineID])
            conversations.append(convObj)

    # 生成问答对列表
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation['lines'])-1):
            inputLine = conversation['lines'][i]['text'].strip()
            targetLine = conversation['lines'][i+1]['text'].strip()
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])

    # 将问答对列表保存成txt文件
    delimiter = '\t' # 分隔符

    print('Writing to newly formatted file...')
    with open(data_file, 'w', encoding='utf-8') as outputFile:
        writer = csv.writer(outputFile, delimiter=delimiter)
        for pair in qa_pairs:
            writer.writerow(pair)
    print('Done writing to file, saved as formatted_movie_lines.txt\n')

def save_xiaohuangji_formatted(data_file):
    print("Processing xiaohuangji50w_fenciA.conv...\n")
    conv_path = os.path.join(DATASET_PATH, 'xiaohuangji/xiaohuangji50w_fenciA.conv')
    pairs = []
    with open(conv_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f.readlines() if line != 'E\n']
        # print(content)
    for q, a in zip(lines[::2], lines[1::2]):
        pairs.append([q.strip()[2:].replace('/',' '), a.strip()[2:].replace('/',' ')])

    delimiter = '\t'

    print('Writing to newly formatted file...')
    with open(data_file, 'w', encoding='utf-8') as outputFile:
        writer = csv.writer(outputFile, delimiter=delimiter)
        for pair in pairs:
            writer.writerow(pair)
    print('Done writing to file, saved as formatted_movie_lines.txt\n')

def get_cornell_pairs():
    '''用于获取问答对列表'''
    data_file = os.path.join(DATASET_PATH, 'formatted_movie_lines.txt')
    if not os.path.exists(data_file):
        save_cornell_formatted(data_file)

    print('Reading and processing formatted_movie_lines.txt...')
    pairs = open(data_file, encoding='utf-8').read().strip().split('\n')
    # 标准化文本
    pairs = [[normalizeString(s) for s in pair.split('\t')] for pair in pairs]
    # 过滤空对话
    pairs = [pair for pair in pairs if len(pair)>1]
    # 过滤过长语句
    print('Total {} pairs'.format(len(pairs)))
    pairs = fileterPairs(pairs)
    print('Remain {} pairs'.format(len(pairs)))
    print('Done\n')

    return pairs

def get_xiaohuangji_pairs():
    '''用于获取问答对列表'''
    data_file = os.path.join(DATASET_PATH, 'formatted_xiaohuangji.txt')
    if not os.path.exists(data_file):
        save_xiaohuangji_formatted(data_file)

    print('Reading and processing formatted_xiaohuangji.txt...')
    pairs = open(data_file, encoding='utf-8').read().strip().split('\n')

    pairs = [[s for s in pair.split('\t')] for pair in pairs]

    # 过滤空对话
    pairs = [pair for pair in pairs if len(pair)>1]
    # 过滤过长语句
    print('Total {} pairs'.format(len(pairs)))
    pairs = fileterPairs(pairs)
    print('Remain {} pairs'.format(len(pairs)))
    print('Done\n')

    return pairs

def get_pairs(name):
    if name == 'cornell':
        return get_cornell_pairs()
    elif name == 'xiaohuangji':
        return get_xiaohuangji_pairs()
    else:
        raise('Dataset not found')

if __name__ == "__main__":
    pairs = get_pairs('xiaohuangji')
    print(pairs)

