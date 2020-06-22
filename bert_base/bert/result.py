# encoding: utf-8
"""
@author: sunkai
@contact: moguizhiz@126.com
@time: 2020/4/29 17:20
@file: result.py
@desc: 将推测的结果存入到指定的文件
"""

import re
import pandas as pd
import itertools
import codecs

class Result(object):

    def __init__(self, tokens, tags, line, result_file, rule_file, labels_file):

        self.tokens = tokens
        self.tags = tags
        self.line = line
        self.result_file = result_file
        self.rule_file = rule_file
        self.labels_file = labels_file

    def preprocess(self):
        if len(self.tokens) > len(self.tags):
            self.tokens = self.tokens[:len(self.tags)]

    def write_to_file(self):
        entity_list = []
        frame_list = []
        temp_list = []

        nrows = self.get_labels_num(self.labels_file)
        rule = self.get_rule_matrix(self.rule_file, nrows)

        result_map = self.get_reslut()
        for lable, entities in result_map.items():
            if entities:
               for entity in entities:
                   temp_couple = (entity, lable)
                   entity_list.append(temp_couple)

        entity_couple_list = list((itertools.permutations(entity_list, 2)))

        for temp_couple in entity_couple_list:
            temp_entity01 = temp_couple[0]
            temp_entity02 = temp_couple[1]

            if self.relation_rule_invalid(label1 = temp_entity01[1], label2 = temp_entity02[1], rule = rule):
                print(str(temp_entity01[1]) + " and " + str(temp_entity02[1]) + " is incompatible rule")
                continue

            temp_str01 = "".join(temp_entity01[0]).replace("##","")
            temp_str02 = "".join(temp_entity02[0]).replace("##","")

            temp_str01 = temp_str01.strip()
            temp_str02 = temp_str02.strip()

            if temp_str01 == '[UNK]' or temp_str02 == '[UNK]':
                continue

            temp_str01 = self.replace_UNK(temp_str01,self.line)
            temp_str02 = self.replace_UNK(temp_str02,self.line)

            if self.line.find(temp_str01) == -1 or self.line.find(temp_str01) == -1:
                continue

            temp_list.append(temp_str01)
            temp_list.append(temp_str02)
            temp_list.append(" ")
            temp_list.append(self.line)
            frame_list.append(temp_list)
            temp_list = []
        df = pd.DataFrame(frame_list, columns=list('ABCD'))
        df.to_csv(self.result_file, index=False, header=False, sep=",", encoding="utf_8_sig", mode="a")

    def get_reslut(self):
        entity_name = ""
        last_type = ""
        labels_map = self.init_labels_map()
        for index in range(len(self.tokens)):
            token = self.tokens[index]
            tag = self.tags[index]
            gussed, guessed_type = self.parse_tag(tag)
            if gussed == 'B':
                if entity_name != "":
                    labels_map[last_type].append(entity_name)
                    entity_name = ""
                last_type = guessed_type
                entity_name += token
            elif gussed == 'I' and guessed_type == last_type:
                entity_name += token
            else:
                if entity_name != "":
                    labels_map[last_type].append(entity_name)
                    last_type = ""
                entity_name = ""

        if entity_name != "":
            labels_map[last_type].append(entity_name)

        return  labels_map

    def init_labels_map(self):
        category = set()
        labels_map = {}

        for tag in self.tags:
            _, guessed_type = self.parse_tag(tag)
            if len(guessed_type) == 0:
                continue
            category.add(guessed_type)

        for value in category:
            labels_map[value] = []

        return labels_map

    def parse_tag(self,t):
        m = re.match(r'^([^-]*)-(.*)$', t)
        return m.groups() if m else (t, '')

    def relation_rule_invalid(self, label1, label2, rule):
        try:
            if rule[label1][label2]:
                return False
            return True
        except KeyError as e:
            return True

    def get_rule_matrix(self, filepath, nrows):
        df = pd.read_csv(filepath, header=0, encoding="gbk", index_col=0, nrows=nrows)
        return df

    def get_labels_num(self, file):
        input_data = codecs.open(file, 'r', 'utf-8')
        num = 0
        for line in input_data.readlines():
            line = line.strip()
            word = line.split('-')
            if word[0] == 'B':
                num = num + 1
        input_data.close()
        return num

    def replace_UNK(self, str, line):
        is_find = str.find("[UNK]")
        temp_str = str

        try:
            if is_find != -1:
                str = str.replace(".", "\.")
                str = str.replace("*", "\*")
                str = str.replace(")", "\)")
                str = str.replace("(", "\(")
                str = str.replace("+", "\+")
                regex_str = str.replace("[UNK]", ".")
                temp_str = re.findall(regex_str, line)[0]
        except IndexError as e:
            return str

        return temp_str



