#!/usr/bin/python
# -*- coding: UTF-8 -*-

import codecs
import argparse
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from bert_base.train.common import LABELS_DATA

label_map = {}
no_statistic_list = []

def origin2tag(src_file, dst_file):
    input_data = codecs.open(src_file, 'r', 'utf-8')
    output_data = codecs.open(dst_file, 'w', 'utf-8')
    for line in input_data.readlines():
        line = line.strip()
        if len(line) == 0:
            continue

        i = 0
        while i < len(line):
            if line[i] == '{':
                i += 2
                temp = ""
                # print("line:" + str(line))
                while line[i] != '}':
                    temp += line[i]
                    i += 1
                i += 2
                word = temp.split(':')
                sen = word[1]
                sen = strip_space(sen)
                output_data.write(sen[0] + " B-" + word[0].strip() + " ")
                output_data.write('\n')
                for j in sen[1:len(sen)]:
                    output_data.write(j + " I-" + word[0].strip() + " ")
                    output_data.write('\n')
            else:
                output_data.write(line[i] + " O ")
                output_data.write('\n')
                i += 1
        output_data.write('\n')
    input_data.close()
    output_data.close()


def strip_space(sentence):
    temp_sentence = []
    for i in sentence[0:len(sentence)]:
        i = i.strip()
        if len(i) > 0:
            temp_sentence.append(i)
    return temp_sentence

def cut_sentence(file, max_seq_length):
    """
    句子截断
    :param file:
    :param max_seq_length:
    :return:
    """
    context = []
    sentence = []
    cnt = 0
    for line in load_file(file):
        line = line.strip()

        if line == '' and len(sentence) != 0:
            # 判断这一句是否超过最大长度
            if len(sentence) > max_seq_length:
                sentence = _cut(sentence)
                context.extend(sentence)
            else:
                context.append(sentence)
            sentence = []
            continue
        cnt += 1
        sentence.append(line)
    print('token cnt:{}'.format(cnt))
    return context

def load_file(file_path):
    if not os.path.exists(file_path):
        return None
    with codecs.open(file_path, 'r', encoding='utf-8') as fd:
        for line in fd:
            yield line


def _cut(sentence):
    new_sentence = []
    sen = []
    for i in sentence:
        if i.split(' ')[0] in ['。', '！', '？','\t'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append(sen)
            sen = []
            continue
        sen.append(i)

    if len(new_sentence) == 0: #娄底那种一句话超过max_seq_length的且没有句号的，用,分割，再长的不考虑了。。。
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，'] and len(sen) != 0:
                sen.append(i)
                new_sentence.append(sen)
                sen = []
                continue
            sen.append(i)
    return new_sentence

def write_to_file(file, context):
    # 首先将源文件改名为新文件名，避免覆盖
    with codecs.open(file, 'w', encoding='utf-8') as fd:
        for sen in context:
            for token in sen:
                fd.write(token + '\n')
            fd.write('\n')

def cut_pro(src_file, dst_file, max_seq_length):

    print('cut data to max sequence length:{}'.format(max_seq_length))
    context = cut_sentence(src_file, args.max_seq_length)
    write_to_file(dst_file, context)

def statistics(file):

    for key, _ in label_map.items():
        label_map[key] = 0

    input_data = codecs.open(file, 'r', 'utf-8')
    for line in input_data.readlines():
        line = line.strip()
        # print("statistics:" + str(line))
        word = line.split(' ')
        if len(word) < 2 or word[1] in no_statistic_list:
            continue
        sen = word[1]
        num = label_map[sen]
        num = num + 1
        label_map[sen] = num
    input_data.close()

    total_num = 0
    for _, value in label_map.items():
        total_num = total_num + value

    print(file)
    print("total_num:" + str(total_num))
    for key, value in label_map.items():
        print(key + ': %6.2f%%; ' % (100. * value / total_num))


def update_label_map(file):
    temp_label_map = {}
    input_data = codecs.open(file, 'r', 'utf-8')
    for line in input_data.readlines():
        line = line.strip()
        word = line.split('-')
        if word[0] == 'B':
            temp_label_map[line] = 0
    input_data.close()
    return temp_label_map


def update_no_statistic_list(file):
    temp_list = []
    input_data = codecs.open(file, 'r', 'utf-8')
    for line in input_data.readlines():
        line = line.strip()
        word = line.split('-')
        if word[0] != 'B':
            temp_list.append(line)
    input_data.close()

    return temp_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data pre process')
    parser.add_argument('--train_data', type=str, default='train_wordtag.txt')
    parser.add_argument('--dev_data', type=str, default='dev_wordtag.txt')
    parser.add_argument('--test_data', type=str, default='test_wordtag.txt')
    parser.add_argument('--max_seq_length', type=int, default=126)
    args = parser.parse_args()

    dst_train_data = '../NERdata/train_data'
    dst_dev_data = '../NERdata/dev_data'
    dst_test_data = '../NERdata/test_data'

    data_convert = {
        'train_data.txt': args.train_data,
        'dev_data.txt': args.dev_data,
        'test_data.txt': args.test_data,
    }

    data_cut_convert = {
        args.train_data: dst_train_data,
        args.dev_data: dst_dev_data,
        args.test_data: dst_test_data,
    }

    for key, value in data_convert.items():
        origin2tag(key,value)

    for key, value in data_cut_convert.items():
        cut_pro(key, value, args.max_seq_length)

    label_map = update_label_map(LABELS_DATA)
    no_statistic_list = update_no_statistic_list(LABELS_DATA)

    for _, value in data_convert.items():
        statistics(value)


