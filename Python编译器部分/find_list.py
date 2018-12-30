#coding=utf-8

import os

train_path = './Leaf/train/'
test_path = './Leaf/test/'
save_train='train.txt'
save_test='test.txt'
file_train=open(save_train, 'w')
# file_test=open(save_test, 'w')
result = []#所有的文件
class_dic= {'fengye':'0', 'hehuaye':'1', 'hupilan':'2', 'sangshuye':'3', 'songshuye':'4', 'yushuye':'5', 'zhaocaishu':'6', 'zhongshuye':'7'}

for maindir, subdir, file_name_list in os.walk(train_path):


    class_name= maindir.split('/')[-1]

    for filename in file_name_list:

        output = class_name +'/'+filename+'>'+ class_dic[class_name]
        print(output)
        file_train.write(output +'\n')
