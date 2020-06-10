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

class Result(object):

    def __init__(self, tokens, tags, line, file):

        self.tokens = tokens
        self.tags = tags
        self.file = file
        self.line = line

    def preprocess(self):
        if len(self.tokens) > len(self.tags):
            self.tokens = self.tokens[:len(self.tags)]

    def write_to_file(self):
        entity_list = []
        frame_list = []
        temp_list = []
        result_map = self.get_reslut()
        for _, value in result_map.items():
            entity_list.extend(value)
        entity_couple_list = list((itertools.permutations(entity_list, 2)))
        for temp_couple in entity_couple_list:
            for temp in temp_couple:
                temp_list.append(temp)
            temp_list.append(" ")
            temp_list.append(self.line)
            frame_list.append(temp_list)
            temp_list = []
        df = pd.DataFrame(frame_list, columns=list('ABCD'))
        df.to_csv(self.file, index=False, header=False, sep=",", encoding="utf_8_sig", mode="a")

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


