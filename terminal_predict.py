# encoding=utf-8

"""
基于命令行的在线预测方法
@Author: Macan (ma_cancan@163.com) 
"""

import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
from datetime import datetime
import platform

from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser
from bert_base.bert.result import Result
from bert_base.train.common import LABELS_DATA
from data_preprocess.data_util import _cut


args = get_args_parser()

temp_model_dir = os.path.join("output", "")
temp_bert_dir = os.path.join("chinese_L-12_H-768_A-12", "")
model_dir = temp_model_dir
bert_dir = temp_bert_dir
system = platform.system()
if system == "Linux":
    model_dir = temp_model_dir.replace('\\', '/')
    bert_dir = temp_bert_dir.replace('\\', '/')


is_training=False
use_one_hot_embeddings=False
batch_size=1

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None


print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
    print("label_list length is " + str(len(label_list)))
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, num_labels=num_labels, use_one_hot_embeddings=False, lstm_size=args.lstm_size, dropout_rate=1.0,
        crf_only=args.crf_only, is_add_self_attention=args.is_add_self_attention)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))


tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)


def predict_online(sentence):
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(line, args.max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids],(batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, args.max_seq_length))
        return input_ids, input_mask

    global graph
    with graph.as_default():
        print('input the test sentence:')
        temp_sen = tokenizer.basic_tokenizer.get_origin_tokens(sentence)
        temp_sen = "".join(temp_sen)
        start = datetime.now()
        if len(sentence) < 2:
            print("invalid sentence:" + sentence)
            return
        sentence = tokenizer.tokenize(sentence)
        print('your input is:{}'.format(sentence))
        input_ids, input_mask = convert(sentence)

        feed_dict = {input_ids_p: input_ids,
                       input_mask_p: input_mask}
        pred_ids_result = sess.run([pred_ids], feed_dict)
        pred_label_result = convert_id_to_label(pred_ids_result, id2label)

        print("tags:" + str(pred_label_result[0]))

        result = Result(sentence,pred_label_result[0],temp_sen,args.result_file, args.rule_file, os.path.join(args.data_prepro_dir, LABELS_DATA))
        result.preprocess()
        result.write_to_file()

        print('time used: {} sec'.format((datetime.now() - start).total_seconds()))

def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result


def convert_single_example(example, max_seq_length, tokenizer):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param example: 一个样本
    :param max_seq_length:
    :param tokenizer:
    :return:
    """

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    # append("O") or append("[CLS]") not sure!
    for i, token in enumerate(tokens):
        ntokens.append(token)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    # append("O") or append("[SEP]") not sure!
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        # we don't concerned about it!
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=None,
        label_ids=None,
        # label_mask = label_mask
    )
    return feature

if __name__ == "__main__":

    if os.path.exists(args.result_file):
        os.remove(args.result_file)

    input_data = codecs.open(args.pre_file, 'r', 'utf-8')
    for line in input_data.readlines():
        line = line.strip()
        sentence = _cut(line)

        for i in sentence:
            temp_line = "".join(i)
            temp_line = temp_line.strip()
            if len(temp_line) != 0:
                predict_online(temp_line)
    input_data.close()


