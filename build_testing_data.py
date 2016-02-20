# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os

def write2file(data, dst_file, idx_begin, idx_end):
    i = idx_begin
    while i < idx_end:
        dst_file.write(data[i])
        i += 1
    dst_file.write("\n")

def build_testing_data():
    root_dir = "/gds/zhwang/zhwang/data/cuhk/";
    src_dir = root_dir + "st/"
    training_path = root_dir + "training_data"
    testing_path = root_dir + "testing_data"
    
    with open(training_path, "w") as f_training:
        with open(testing_path, "w") as f_testing:
            for ld in os.listdir(src_dir):
                src_path = os.path.join(src_dir, ld)
                data = []
                with open(src_path, "r") as f_src:
                    while 1:
                        l = f_src.readline()
                        if not l:
                            break
                        data.append(l)
                idx_begin = 0
                idx_end = 0
                while 1:
                    while data[idx_end] != "\n" and idx_end < len(data):
                        idx_end += 1
                    if data[idx_end + 1] == "----\n":
                        write2file(data, f_testing, idx_begin, idx_end)
                        if idx_end + 2 >= len(data):
                            break
                        idx_begin = idx_end + 2
                        idx_end = idx_begin
                        while data[idx_end] != "\n":
                            idx_end += 1
                        write2file(data, f_testing, idx_begin, idx_end)
                    else:
                        write2file(data, f_training, idx_begin, idx_end)
                    idx_end += 1
                    idx_begin = idx_end

build_testing_data()

