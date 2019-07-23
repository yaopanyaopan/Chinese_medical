# -*-coding:utf-8 -*-
__author__ = '$'

import csv
import numpy as np
import re
import math
import xlrd
import xlwt
import pandas as pd
import operator
import convert_shuzi
import os
import random
import jieba
from gensim import *
import gensim
import sys

csv.field_size_limit(sys.maxsize)


def bieming():
    BM_func = {}
    with open('data/功效别名.csv', 'r') as fr:
            reader = csv.reader(fr)
            for hhh in reader:
                ss = hhh[1].strip().split(' ')
                BM_func.update({hhh[0]: ss})

    BM = {}
    with open('data/中药别名.csv', 'r') as fr:
        reader = csv.reader(fr)
        for hhh in reader:
            ss = hhh[1].strip().split(' ')
            BM.update({hhh[0]: ss})

    w = xlrd.open_workbook('data/final.xls')
    sheet = w.sheets()[0]
    rows = sheet.nrows
    all = []
    for i in range(rows):
        # print(sheet.row_values(i))
        medical = sheet.cell_value(i,0).strip()
        func = sheet.cell_value(i,1).strip()
        zhuzhi = sheet.cell_value(i,2).strip()

        ms = medical.split(',')
        res = []
        # print(ms)
        for med in ms:
            res.append(med)
            for k,v in BM.items():
                if med in v:
                    res[-1] = k
                    break
        res = ','.join(res)

        for k,v in BM_func.items():
            for val in v:
                if val in func:
                    func = func.replace(val,k)
                    break
        all.append([res,func,zhuzhi])
    w = xlwt.Workbook(encoding='utf8')
    ws = w.add_sheet('name')
    for i in range(len(all)):
        for j in range(len(all[i])):
            ws.write(i,j,all[i][j])
    w.save('data/final_last.xls')



def word2vec():

    stop_list = []
    fr2 = open('stop_words.txt',encoding='utf8')
    for i in fr2:
        if len(i.strip())>0:
            stop_list.append(i)
    sentences = []
    with open('func_5w_addresult.csv','r') as fr:
        reader = csv.reader(fr)
        for line in reader:
            sentence = line[2].split(',')
            res = []
            # print(sentence)
            for ss in range(0,len(sentence),2):
                if sentence[ss] not in stop_list:
                    res.append(sentence[ss])
            if len(res)>0:
                # print(res)
                sentences.append(res)
    # print(sentences)
    w = xlrd.open_workbook('训练药向量语料.xlsx')
    sheet = w.sheets()[0]
    lines1 = sheet.col_values(0)
    # print(lines1)
    for line in lines1:
        sentence = line.split(',')
        res = []
        # print(sentence)
        for ss in sentence:
            if ss not in stop_list:
                res.append(ss)
        if len(res) > 0:
            sentences.append(res)
    # print(sentences)
    print('开始训练...')
    model = gensim.models.Word2Vec(sentences,size=50,window=5,min_count=2,workers=10)
    print(model.wv.word_vec('甘草'))
    print(model.wv.most_similar('甘草'))
    print(model.wv.index2entity)

    model.save('wv50.model')

    print('开始载入...')
    Model = gensim.models.Word2Vec.load('wv50.model')
    model.wv.save_word2vec_format('Mwv50.model.bin',binary=True)


def quchong1():

    with open('data/data5W_new.csv','w') as fw:   # 易红添加的数据
        writer = csv.writer(fw)
        writer.writerow([ '药方名称','别名','处方','制法或炮制','功能主治','用法用量','注意','摘录','各家论述','性状','规格','贮藏'])
        w = xlrd.open_workbook('data/原总数据10.21.xlsx')
        sheet = w.sheets()[0]
        rows = sheet.nrows
        for i in range(rows):               # 易红添加的数据
            val0 = sheet.cell_value(i, 0)  # 药方名
            val1 = sheet.cell_value(i, 1)  # 别名
            val2 = sheet.cell_value(i, 2)  # 处方药组
            val3 = sheet.cell_value(i, 3)  # 制法
            val4 = sheet.cell_value(i, 4)  # 功效
            val5 = sheet.cell_value(i, 5)  # 用量
            val6 = sheet.cell_value(i, 6)  # 注意
            val7 = sheet.cell_value(i, 7)  # 摘录
            val8 = sheet.cell_value(i, 8)  # 各家论述
            val9 = sheet.cell_value(i, 9)  # 性状
            val10 = sheet.cell_value(i, 10)  # 规格
            val11 = sheet.cell_value(i, 11)  # 贮藏

            print(val0,val1,val2)
            writer.writerow([val0,val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val10,val11])


def doit():

    dict = {}
    f = open('func_5w_result.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['药方名称', '别名', '处方', '制法或炮制', '功效主治','整理的方剂功效','未找到方剂功效', '用法用量', '注意', '摘录', '各家论述', '性状', '规格', '贮藏'])
    p_zhi4 = re.compile(r'(.+?)(，|。)(.*)')  # *****‘治气补劳’这类4字开头的算作功效********(非贪婪)

    p12 = re.compile(
        r'(兼能[治]|现用[于]|用[于]|适用[于]|[治]|兼[治]|能[治]|并[治]|亦[治]|又[治]|及[治]|主[治]|专[治]|适用)(.+?)(。|，|；| )([主]|并[主])(.+)')  # ‘治...主。。。‘
    p1 = re.compile(r'(兼能[治]|现用[于]|用[于]|适用[于]|[治]|兼[治]|能[治]|并[治]|亦[治]|又[治]|及[治]|主[治]|专[治]|适用)(.+)')  # '治...'
    p01 = re.compile(
        r'(.+?)(，|。|；| )(兼能[治]|现用[于]|用[于]|适用[于]|[治]|兼[治]|能[治]|并[治]|亦[治]|又[治]|及[治]|主[治]|专[治]|适用)(.+)')  # .....。治...
    p02 = re.compile(r'(.*?)(。|，|；| )([主]|并[主])(.+)')  # .....。主。。。
    p2 = re.compile(r'([主]|并[主])(.+)')  # 主.....
    p21 = re.compile(
        r'([主]|并[主])(.+?)(，|。|；| )(兼能[治]|现用[于]|用[于]|适用[于]|[治]|兼[治]|能[治]|并[治]|亦[治]|又[治]|及[治]|主[治]|专[治]|适用)(.*)')  # 主...治...
    p30 = re.compile(r'(.*?)(。|，|；| )([主]|并[主])')

    with open('data/data5W_new.csv', 'r') as f:
        reader = csv.reader(f)
        count = 0
        for line in reader:
            if count==0:
                print(line)
                count += 1
                continue
            res = []
            for kkk in line:
                res.append(kkk)
            count +=1
            L = []       # s[1]为前面的数字序号，s[2]为文字内容
            str1 = line[4].strip()
            if p12.match(str1):      # 治...主...
                m = p12.match(str1)
                # print(m.groups())
                r = ' '+m.group(1) + m.group(2)
                L.append(r)
                # print(L)
            elif p1.match(str1):     # 治.....
                m = p1.match(str1)
                # print(m.groups())
                r =  ' '
                L.append(r)
                # print(L)
                if '主' in L:
                    print(L)
            elif p21.match(str1):     # 主....治....
                m = p21.match(str1)
                # print(m.groups())
                r =  ' '
                L.append(r)
                # print(L)
            elif p2.match(str1):  # 主.....
                m = p2.match(str1)
                # print(m.groups())
                r =  ' '
                L.append(r)
                # print(L)
                if '治' in L:
                    print(L)
            elif p01.match(str1):    # ....治....
                m = p01.match(str1)
                str1 = m.group(1)
                str4 = m.group(4)
                if p01.match(str1):    # ...治。。。
                    m1 = p01.match(str1)
                    # print(m1.groups())
                    str11 = m1.group(1)
                    if p01.match(str11):  # ...治。。。
                        m11 = p01.match(str11)
                        # print(m11.groups())
                        if p02.match(m11.group(1)):  # ...主。。。
                            m112 = p02.match(m11.group(1))
                            # print(m112.groups())
                            r = ' ' + m112.group(4)
                            L.append(r)
                            # print(L)
                        if p01.match(m11.group(1)):  # 。。。治。。。
                            m111 = p02.match(m11.group(1))
                            # print(m111.groups())
                            r = ' ' + m111.group(1)
                            L.append(r)
                           # print(L)
                elif p02.match(str1):    # ...主。。。
                    # print(m.groups())
                    m = p02.match(str1)
                    r = ' ' + m.group(1) + '，'
                    L.append(r)
                    # print(L)
                else:
                    r = ' ' + str1 + '，'
                    L.append(r)
                    # print(m.groups())
                    # print(L)
                if p02.match(str4):  # ...主。。。
                    m = p02.match(str4)
                    pass
                    # print(m.groups())
                    # r = ' '
                    # L.append(r)
                    # print(L)
                elif p01.match(str4):  # ...治。。。
                    m = p01.match(str4)
                    pass
                    # print(m.groups())
                    # r = ' '
                    # L.append(r)
                    # print(L)
                else:
                    pass
                    r = ' '
                    # L.append(r)
                   # print(L)
            elif p02.match(str1):  # ...主。。。
                m = p02.match(str1)
                str1 = m.group(1)
                str4 = m.group(4)
                if p02.match(str1):  # ...主。。。
                    m02 = p02.match(str1)
                    r = ' ' + m02.group(1) + '，' + m02.group(4) + '，'
                    L.append(r)
                    # print(L)
                else:
                    r = ' ' + str1 + '，'
                    L.append(r)
                    # print(m.groups())
                    # print(L)
                if p01.match(str4):  # ...治。。。
                    m01 = p01.match(str4)
                    r = ' ' + m01.group(1) + '，'
                    L.append(r)
                    # print(m.groups())
                    # print(L)
                else:
                    pass
                    # r = ' '
                    # L.append(r)
                    # print(m.groups())
                    # print(L)
            else:    # 不含 主 或 治
                if p30.match(str1):
                    m = p30.match(str1)
                    # print(m.groups())
                    r = ' ' + m.group(1) + '，'
                    L.append(r)
                    # print(L)
                else:
                    L.append(' ')
                    r = str1
                    L.append(r)
                    # print(L)
            res1 = []
            for dd in range(len(res)+2):
                # if dd==4:
                #     res1.append()
                if dd==5:
                    res1.append(L[0])
                if dd==6:
                    if len(L)>1:
                        # print(L[1])
                        print(res)
                        res1.append(L[1])
                    else:
                        res1.append(' ')
                if dd<5:
                    res1.append(res[dd])
                if dd>6:
                    res1.append(res[dd-2])
            writer.writerow(res1)

def add_fangjis():
    with open('func_5w_add.csv','w') as fw:
        writer = csv.writer(fw)
        fr = open('func_5w_result.csv', 'r')
        reader = csv.reader(fr)
        for i in reader:
            writer.writerow(i)
        pattern = re.compile(r'(?:、|，|:|：|；|。|,)')
        pattern11 = re.compile(
            r'\d+.\d+(?:g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)')
        pattern22 = re.compile(
            r'\d+(?:g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)')
        pattern111 = re.compile(
            r'(\d+.\d+(?:g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔))|(\d+(?:g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔))|(\d+.\d+)|(\d+)')

        w = xlrd.open_workbook('data/add_fangjis/fangji.xlsx')
        sheet = w.sheets()[0]
        nrows = sheet.nrows
        for i in range(1,nrows):
            res = []
            values = sheet.row_values(i)
            # print(values[3])   # 方剂名
            # print(values[8])   # 功效
            # print(values[10])   # 用法剂量
            # print(values[20])   # 药组
            res.append(values[3])
            res.append('')
            resss = re.sub(pattern, ' ', values[20])
            resss = resss.replace('(', '')
            resss = resss.replace(')', '')
            resss = resss.replace('（', '')
            resss = resss.replace('）', '')
            if re.split(pattern111, resss):
                resss = pattern111.split(resss)
                ss = []
                for j in resss:
                    try:
                        if isinstance(j,str) and j!='':
                            jiliang = float(j)
                            jiliang = str(jiliang)+'g'
                            ss.append(jiliang)
                    except:
                        # print(j)
                        if isinstance(j,str) and j!='':
                            ss.append(j)
            else:
                print(resss)
            # print(ss)
            for now in range(0,len(ss)-1):
                if pattern11.findall(ss[now]) or pattern22.findall(ss[now]):
                    check11 = False
                else:
                    check11 = True
                if pattern11.findall(ss[now+1]) or pattern22.findall(ss[now+1]):
                    check22 = True
                else:
                    check22 = False
                if check11 and check22:
                    hb = ss[now]+ss[now+1]
                    ss[now] = hb
                    ss[now+1] = ''
            sss = []
            for dd in ss:
                if dd.strip()!='':
                    sss.append(dd.strip())
            # print(sss)
            sss = ' '.join(sss)
            if '上方' == sss.strip():
                continue
            if '1方' == sss:
                continue
            res.append(sss)
            res.append('')
            res.append(values[8])
            res.append(values[8])
            res.append('')
            res.append(values[10])
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            writer.writerow(res)

        w = xlrd.open_workbook('data/add_fangjis/小记录.xlsx')
        sheet = w.sheets()[0]
        nrows = sheet.nrows
        pp = re.compile(r'(\d+)剂|付|副')
        ppp1 = re.compile(r'(=\d+/\d+/\d+)')   #去掉       =50/3/70
        ppp2 = re.compile(r'(=\d+/\d+)')       #去掉药组中  =3 / 70
        ppp3 = re.compile(r'(\d+/\d+)')
        ppp4 = re.compile(r'(=\d+g /\d+/\d+g)')
        ppp41 = re.compile(r'=/\d+g /g')
        ppp5 = re.compile(r'(/\d+g)')
        ppp6 = re.compile(r'(/\d+)')
        ppp7 = re.compile(r'=/\d+g')
        ppp8 = re.compile(r'/g')
        pppp = re.compile(r'(\d+方)')
        for i in range(1, nrows):
            res = []
            values = sheet.row_values(i)

            # print(values[14])   # 方剂名
            # print(values[13])   # 功效
            # print(values[15])   # 药组
            res.append(values[14])
            res.append('')
            resss = re.sub(pattern, ' ', values[15])
            resss = resss.replace('(', '')
            resss = resss.replace(')', '')
            resss = resss.replace('（', '')
            resss = resss.replace('）', '')
            resss = resss.replace('+', '')
            resss = resss.replace('-', '')
            resss = pp.sub('',resss)  # 去掉 ‘3剂’
            # print(resss)
            if re.split(pattern111, resss):
                resss = pattern111.split(resss)
                ss = []
                for j in resss:
                    try:
                        if isinstance(j, str) and j != '':
                            jiliang = float(j)
                            jiliang = str(jiliang) + 'g'
                            ss.append(jiliang)
                    except:
                        # print(j)
                        if isinstance(j, str) and j != '':
                            ss.append(j)
            else:
                print(resss)
            # print(ss)
            for now in range(0, len(ss) - 1):
                if pattern11.findall(ss[now]) or pattern22.findall(ss[now]):
                    check11 = False
                else:
                    check11 = True
                if pattern11.findall(ss[now + 1]) or pattern22.findall(ss[now + 1]):
                    check22 = True
                else:
                    check22 = False
                if check11 and check22:
                    hb = ss[now] + ss[now + 1]
                    ss[now] = hb
                    ss[now + 1] = ''
            sss = []
            for dd in ss:
                if dd.strip() != '':
                    sss.append(dd.strip())
            # print(sss)
            sss = ' '.join(sss)
            sss = sss.replace('/外洗','')
            if '上方' == sss:
                continue
            if '有' == sss:
                continue
            if '有药' == sss:
                continue
            if '无' == sss:
                continue
            if '针刺' == sss:
                continue
            sss = sss.replace('上方加', '')
            sss = sss.replace('上方减', '')
            sss = sss.replace('上方', '')
            sss = sss.replace('诊方', '')
            if '1方' == sss:
                continue
            sss = sss.replace('1方', '')
            sss = sss.replace('一诊方', '')
            sss = sss.replace('M', '')
            sss = sss.replace('N', '')
            sss = sss.replace('今天和明天', '')
            sss = sss.replace('今天', '')
            sss = sss.replace('明天', '')
            sss = sss.replace('上午和下午', '')
            sss = sss.replace('上午', '')
            sss = sss.replace('下午', '')
            if sss=='/' or sss=='' or '未服' in sss or '拿到' in sss or '暂停' in sss or '崭停' in sss\
                    or '味道' in sss or '服后' in sss or 'it' in sss or 'IT' in sss or 'STOP' in sss or \
                    'stop' in sss or '泡酒药' in sss or '收到' in sss or 'GET' in sss or '咨询' in sss or 'PAUS' in sss\
                    or '自调' in sss or '耒服' in sss or '少量服' in sss or '医生' in sss or '外出' in sss\
                    or '不舒服' in sss or '呕吐' in sss or'不愿' in sss  or '没有' in sss or '自己' in sss\
                    or '怕针' in sss or '抗流感' in sss or '今订' in sss or '下次' in sss or '没钱' in sss or '电话' in sss\
                    or '服药' in sss or '停药' in sss or 'SOTP' in sss or 'Homopatisch' in sss or '自制' in sss\
                    or '服用' in sss or 'EAT' in sss or 'TUI' in sss or '未完' in sss or '得到' in sss or '开始' in sss\
                    or '渡假' in sss or '问题' in sss or '应加' in sss or '服自然' in sss or '指标' in sss or '天后' in sss\
                    or '=/' in sss:
                continue
            res.append(sss)
            res.append('')
            res.append(values[13])
            res.append(values[13])
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            writer.writerow(res)

        w = xlrd.open_workbook('data/add_fangjis/瑞士病历.xlsx')
        sheet = w.sheets()[0]
        nrows = sheet.nrows

        for i in range(1, nrows):
            res = []
            values = sheet.row_values(i)
            # print(values[14])   # 方剂名
            # print(values[13])   # 功效
            # print(values[15])   # 药组
            # if '上方' in values[14]:
            #     continue

            res.append(values[14])
            res.append('')
            resss = re.sub(pattern, ' ', values[15])
            resss = resss.replace('(', '')
            resss = resss.replace(')', '')
            resss = resss.replace('（', '')
            resss = resss.replace('）', '')
            resss = resss.replace('+', '')
            resss = resss.replace('-', '')
            resss = resss.replace('#', '')
            resss = pp.sub('', resss)  # 去掉 ‘3剂’
            resss = pppp.sub('', ''.join(resss))
            resss = ppp4.sub('', ''.join(resss))
            resss = ppp41.sub('', ''.join(resss))
            resss = ppp1.sub('', ''.join(resss))
            resss = ppp2.sub('', ''.join(resss))
            resss = ppp3.sub('', ''.join(resss))
            resss = ppp5.sub('', ''.join(resss))
            resss = ppp6.sub('', ''.join(resss))
            resss = ppp7.sub('', ''.join(resss))
            resss = ppp8.sub('', ''.join(resss))

            if re.split(pattern111, resss):
                resss = pattern111.split(resss)
                ss = []
                for j in resss:
                    try:
                        if isinstance(j, str) and j != '':
                            jiliang = float(j)
                            jiliang = str(jiliang) + 'g'
                            ss.append(jiliang)
                    except:
                        # print(j)
                        if isinstance(j, str) and j != '':
                            ss.append(j)
            else:
                print(resss)
            # print(ss)
            for now in range(0, len(ss) - 1):
                if pattern11.findall(ss[now]) or pattern22.findall(ss[now]):
                    check11 = False
                else:
                    check11 = True
                if pattern11.findall(ss[now + 1]) or pattern22.findall(ss[now + 1]):
                    check22 = True
                else:
                    check22 = False
                if check11 and check22:
                    hb = ss[now] + ss[now + 1]
                    ss[now] = hb
                    ss[now + 1] = ''
            sss = []
            for dd in ss:
                if dd.strip() != '':
                    sss.append(dd.strip())
            # print(sss)
            sss = ' '.join(sss)
            sss = sss.replace('/外洗','')
            if '上方' == sss:
                continue
            if '有' == sss:
                continue
            if '有药' == sss:
                continue
            if '无' == sss:
                continue
            if '针刺' == sss:
                continue
            sss = sss.replace('上方加', '')
            sss = sss.replace('上方减', '')
            sss = sss.replace('上方', '')
            sss = sss.replace('诊方', '')
            if '1方' == sss:
                continue
            sss = sss.replace('1方', '')
            sss = sss.replace('一诊方', '')
            sss = sss.replace('M', '')
            sss = sss.replace('N', '')
            sss = sss.replace('今天和明天', '')
            sss = sss.replace('今天', '')
            sss = sss.replace('明天', '')
            sss = sss.replace('上午和下午', '')
            sss = sss.replace('上午', '')
            sss = sss.replace('下午', '')
            if sss=='/' or sss=='' or '未服' in sss or '拿到' in sss or '暂停' in sss or '崭停' in sss\
                    or '味道' in sss or '服后' in sss or 'it' in sss or 'IT' in sss or 'STOP' in sss or \
                    'stop' in sss or '泡酒药' in sss or '收到' in sss or 'GET' in sss or '咨询' in sss or 'PAUS' in sss\
                    or '自调' in sss or '耒服' in sss or '少量服' in sss or '医生' in sss or '外出' in sss\
                    or '不舒服' in sss or '呕吐' in sss or'不愿' in sss  or '没有' in sss or '自己' in sss\
                    or '怕针' in sss or '抗流感' in sss or '今订' in sss or '下次' in sss or '没钱' in sss or '电话' in sss\
                    or '服药' in sss or '停药' in sss or 'SOTP' in sss or 'Homopatisch' in sss or '自制' in sss\
                    or '服用' in sss or 'EAT' in sss or 'TUI' in sss or '未完' in sss or '得到' in sss or '开始' in sss\
                    or '渡假' in sss or '问题' in sss or '应加' in sss or '服自然' in sss or '指标' in sss or '天后' in sss\
                    or '=/' in sss:
                continue
            res.append(sss)
            res.append('')
            res.append(values[13])
            res.append(values[13])
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            res.append('')
            writer.writerow(res)


def process_medicals():
    BM_func = {}
    with open('data/功效别名.csv', 'r') as fr:
        reader = csv.reader(fr)
        for hhh in reader:
            ss = hhh[1].split(' ')
            BM_func.update({hhh[0]: ss})

    with open('func_5w_add.csv','r') as fr:
            reader = csv.reader(fr)
            pattern = re.compile(r'(?:、|，|:|：|；|。|,)')
            pattern1 = re.compile(r'(?:\(|（)')
            pattern2 = re.compile(r'(?:\)|）)')
            pattern11 = re.compile(
                r'\d+.\d+(?:g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)')
            pattern22 = re.compile(
                    r'\d+(?:g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)')
            pattern111 = re.compile(
                r'(\d+.\d+(?:g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔))|(\d+(?:g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔))|\
                (\d+.\d+)|(\d+)')
            data_list = []
            nnnn=0
            source_data = []
            for item in reader:
                # print(item[4])
                hang = []
                for hang_data in item:
                    hang.append(hang_data)
                source_data.append(hang)
                nnnn +=1
                if pattern111.split(item[2]):
                    ress = pattern111.split(item[2])
                    # print(ress)
                    rrres = []
                    for ddd in ress:
                        if isinstance(ddd,str):
                            rrres.append(ddd)
                    ress = ' '.join(rrres)
                else:
                    ress = item[2]
                # print(ress)
                res = re.sub(pattern, ' ', ress)
                res = re.sub(pattern1, ' (', res)
                res = re.sub(pattern2, ') ', res)
                res = res.split(' ')
                # print(nnnn,res)
                data_after = []
                for i in res:
                    itemdata = i
                    try:
                        num1,num2 = kuohaoClear(i)
                        itemdata = i[0:num1] + i[num2:]
                        # print(i)
                        content = i[(num1+1):(num2-1)]  # 括号内的内容 , 过滤出剂量大小
                        # print(content)
                        content = convert_shuzi.convert_shuzi(content)
                        weight = ''
                        weight1 = pattern11.findall(content)
                        weight2 = pattern22.findall(content)
                        # 把正确的值放在变量weight中
                        if (weight1):
                            weight = weight1[0]
                            yaowulist = pattern11.split(itemdata)
                            # print(itemdata)
                            # print('w1',weight1)
                            # print(weight)
                        elif (weight2):
                            weight = weight2[0]
                            yaowulist = pattern22.split(itemdata)
                            # print(weight)
                        else:
                            weight=''
                        # print(weight)
                        if len(weight.strip())>0 and len(content)<8: # 括号内的内容少，才可能是剂量
                            data_after.append(weight.strip())
                        else:
                            if len(itemdata.strip())>0 :
                                data_after.append(itemdata.strip())

                    except:
                        if pattern1.search(itemdata):
                            itemdata = itemdata[:pattern1.search(itemdata).start()]
                            if len(itemdata.strip())>0 :
                                data_after.append(itemdata.strip())
                                # print(itemdata)

                        if pattern2.search(itemdata):
                            itemdata = itemdata[pattern2.search(itemdata).end():]
                            if len(itemdata.strip())>0 :
                                data_after.append(itemdata.strip())
                                # print(itemdata)

                        if len(itemdata.strip()) > 0 :
                            data_after.append(itemdata.strip())
                            # print(itemdata)
                # print(data_after)
                data_list.append(data_after)

            medicals_data = extractnumwithstr(data_list)  # 将'人参1千克' 分离成 '人参'，'1千克'
            #------将处理完成的 药组-剂量 存入-------
            fw = open('func_5w_addresult.csv','w')
            writer = csv.writer(fw)
            all_medicals = []
            for ite in range(len(medicals_data)):
                line = []
                for line_item in medicals_data[ite]:
                    for other_data in line_item:
                        line.append(other_data)

                source_data[ite][2] = ','.join(line)
                func = source_data[ite][4]
                for k,v in BM_func.items():
                    for val in v:
                        if val in func :
                            func= func.replace(val , k)    # 清洗 功效别名
                            # print(val,'---',k)
                source_data[ite][4] = func

            writer.writerows(source_data)


#清除方剂里的括号里的补充说明
def kuohaoClear(str):
    # print 'kuohaoClear'
    pattern1 = re.compile(r'(?:\(|（)')
    pattern2 = re.compile(r'(?:\)|）)')
    num1=pattern1.search(str).start()
    num2=pattern2.search(str).end()

    return num1,num2
def extractnumwithstr(data_list):   # 将'人参1千克' 分离成 '人参'，'1千克'
    i = 1
    #(?:..):(...)的不分组版本，用于使用| 或 后接数量词
    pattern1 = re.compile(
        r'\d+.\d+(?:g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)')
    pattern2 = re.compile(
        r'\d+(?:g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)')
    finalmedicallist = []

    BM = {}
    with open('data/中药别名.csv','r') as fr:
        reader = csv.reader(fr)
        for hhh in reader:
            ss = hhh[1].split(' ')
            BM.update({hhh[0]:ss})
    SP = []
    sp = xlrd.open_workbook('data/特殊剂量表.xlsx')
    sheet =sp.sheets()[0]
    rows = sheet.nrows
    for i in range(1,rows):
        SP.append(sheet.row_values(i))

    stop_t = set()
    with open('data/stop.txt','r') as fw:
        texts = fw.readlines()
        for text in texts:
           if text.strip()!='':
               stop_t.add(text.strip())
    for item in data_list:

        medicallist = []
        point = []
        medicaldict = []
        # print(item)
        for itemdata in item:
            weight = ''
            yaowulist = []
            itemdata = convert_shuzi.convert_shuzi(itemdata)   # 一分 ==》 1分
            #在处方内容中通过正则匹配找出数量单位 start
            weight1 = pattern1.findall(itemdata)
            weight2 = pattern2.findall(itemdata)
            #把正确的值放在变量weight中
            if(weight1):
                weight = weight1[0]
                yaowulist = pattern1.split(itemdata)
                # print(itemdata)
                # print('w1',weight1)
            elif(weight2):
                weight = weight2[0]
                yaowulist = pattern2.split(itemdata)

            else:   # 只有药物

                if len(itemdata.strip())>0:
                    yaowulist.append(itemdata)
                    yaowulist.append('')

            # print(weight)
            # 把处方的每味药提出来重新放在medicallist列表元素[0]里，同时已经去除了药的数量单位
            if(yaowulist):
                try:
                    yaowulist.remove('')
                    # print(yaowulist)
                    # for zz in yaowulist:
                    medicallist.append(' '.join(yaowulist))
                except:
                    medicallist.append('')
                    pass
            else:
                medicallist.append(itemdata)
            # 把处方的每味药所对应的数量单位存入medicallist 列表元素[1]的位置列表里
            if(weight):
                medicallist.append(weight)
            else:
                medicallist.append('None')
            # print(medicallist)
            medicaldict.append(medicallist)
            medicallist = []
        # print(medicaldict)
        j=0
        for k,v in medicaldict:
            #用point记录#在哪味药上
            # print (k,v)
            if(k.find('各')>0):
                point.append(j)
                medicaldict[j][0] = medicaldict[j][0].replace('各','')
                # print ('检测到“各”字，该味药在处方中所处位置：',j,k)
            if (k.find('各')==0):
                point.append(j-1)
                medicaldict[j][0] = medicaldict[j][0].replace('各','')
                medicaldict[j-1][1] = medicaldict[j][1]
            j +=1
        # print ('该方剂一共配药数量为：',j)
        # print ('该方剂中出现“各”字的位置有：', point)
        f = 0
        # print ('medicaldict', medicaldict)
        for now in range(len(medicaldict)-1):
            if medicaldict[now][1]=='None' and medicaldict[now+1][0]=='' and medicaldict[now+1][1]!='None':
                # print('313')
                medicaldict[now][1]=medicaldict[now+1][1]
                medicaldict[now+1][1] = ''
        for now in range(len(medicaldict)-1):
            if medicaldict[now][1] != '' and medicaldict[now][0]!='' and medicaldict[now+1][0]=='' and medicaldict[now+1][1]!='':
                if pattern1.findall(medicaldict[now][1]):
                    continue
                if pattern2.findall(medicaldict[now][1]):
                    continue
                medicaldict[now][1]=medicaldict[now+1][1]
                medicaldict[now+1][1] = ''
        #---------- 各 -------
        for m,n in medicaldict:
            if(point!=[]):
                for pointnum in point:
                    # print 'test', pointnum
                    if(f>pointnum):
                        continue
                    elif (n=='None'):
                            # print 'test4',medicaldict[f][1],pointnum
                            medicaldict[f][1] = medicaldict[pointnum][1]  # 替换 '各5g'
                            break
            f+=1
        # print ('@处理结果:', medicaldict)
        # 整理部分 ['龟版', 'None'], ['', '500克'] 为['龟版','500克']
        for now in range(len(medicaldict)-1):
            if medicaldict[now][1]=='None' and medicaldict[now+1][0]=='' and medicaldict[now+1][1]!='None':
                # print('313')
                medicaldict[now][1]=medicaldict[now+1][1]
                medicaldict[now+1][1]=''

        for now in range(len(medicaldict)-1):
            if medicaldict[now][1]!='' and medicaldict[now][0]!='' and medicaldict[now+1][0]=='' and medicaldict[now+1][1]!='':
                if pattern1.findall(medicaldict[now][1]):
                    continue
                if pattern2.findall(medicaldict[now][1]):
                    continue
                medicaldict[now][1]=medicaldict[now+1][1]
                medicaldict[now+1][1]=''

        # print(medicaldict)
        for now in range(len(medicaldict)):
            # all_med.add(medicaldict[now][0])
            if medicaldict[now][0]=='':
                pass
            else:
                if medicaldict[now][1] == '':
                    medicaldict[now][1] = 'None'
            if medicaldict[now][0] in stop_t:
                # print(medicaldict[now][0])
                medicaldict[now][0] = ''
                medicaldict[now][1] = ''

        # print(i,medicaldict)
        res_medicaldict = []
        for now in range(len(medicaldict)):
            if medicaldict[now][0]!='':
                for ke,va in BM.items():    # 清洗 中药别名
                    if medicaldict[now][0] in va:
                        # print(medicaldict[now][0],'>>>',ke)
                        medicaldict[now][0]=ke
                        break
                if medicaldict[now][0]!='None':   #特殊剂量
                    for hanghang in range(len(SP)):
                        danwei = SP[hanghang][1]
                        if medicaldict[now][0]==SP[hanghang][0] and danwei in medicaldict[now][1]:
                            p1 = re.compile(r'(\d+)(-|～|-|——|~|至)(\d+)'+danwei)
                            p2 = re.compile(r'(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)'+danwei)
                            p3 = re.compile(r'(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)' + danwei)
                            p4 = re.compile(r'(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)' + danwei)
                            if p1.match(medicaldict[now][1]):
                                # print(medicaldict[now][0],medicaldict[now][1])
                                r1 = p1.match(medicaldict[now][1]).groups()
                                medicaldict[now][1] = str((float(r1[0])+float(r1[2]))/2* float(SP[hanghang][2]))+'g'
                                # print(medicaldict[now][0], medicaldict[now][1])
                                break
                            if p2.match(medicaldict[now][1]):
                                # print(medicaldict[now][0], medicaldict[now][1])
                                r2 = p2.match(medicaldict[now][1]).groups()
                                medicaldict[now][1] = str((float(r2[0]+'.'+r2[1])+float(r2[3]))/2* float(SP[hanghang][2]))+'g'
                                # print(medicaldict[now][0], medicaldict[now][1])
                                break
                            if p3.match(medicaldict[now][1]):
                                # print(medicaldict[now][0], medicaldict[now][1])
                                r3 = p3.match(medicaldict[now][1]).groups()
                                medicaldict[now][1] = str((float(r3[2]+'.'+r3[1])+float(r3[0]))/2* float(SP[hanghang][2]))+'g'
                                # print(medicaldict[now][0], medicaldict[now][1])
                                break
                            if p4.match(medicaldict[now][1]):
                                # print(medicaldict[now][0], medicaldict[now][1])
                                r4 = p4.match(medicaldict[now][1]).groups()
                                medicaldict[now][1] = str((float(r4[0]+'.'+r4[1])+float(r4[3]+'.'+r4[4]))/2* float(SP[hanghang][2]))+'g'
                                # print(medicaldict[now][0], medicaldict[now][1])
                                break

                res_medicaldict.append([medicaldict[now][0],medicaldict[now][1]])
        # print(res_medicaldict)
        i +=1
        finalmedicallist.append(res_medicaldict)

    return finalmedicallist


def huafen():

    medicals_gx = []
    x = xlrd.open_workbook('data/最后筛选.xls')
    sheet = x.sheets()[0]
    nrows = sheet.nrows
    for i in range(1,nrows):
        value = sheet.cell_value(i,0)
        medicals_gx.append(value)
    medicals_dict1 = {}    # 标准配伍集里的所有药物
    medicals_dict2 = {}      # 功效对应的 正例
    medicals_dict3 = {}    # 功效对应的 负例  (1.3倍搜集)
    medicals_dict4 = {}     # 评估集
    for i in medicals_gx:
        medicals_dict1.update({i: []})
        medicals_dict2.update({i: []})
        medicals_dict3.update({i: []})
        medicals_dict4.update({i: []})

    x = xlrd.open_workbook('data/final_last.xls')
    sheet = x.sheets()[0]
    nrows = sheet.nrows
    yaozu_lists =[]
    for i in range(1, nrows):
        mm = sheet.cell_value(i, 0)
        gg = sheet.cell_value(i, 1)
        zz = sheet.cell_value(i, 2)
        res =[]
        res.append(mm)
        res.append(gg)
        res.append(zz)
        yaozu_lists.append(res)
    for i in yaozu_lists:
            # print(i[4])
            for medical in medicals_dict1.keys():
                if medical in i[1] :     # 将 功效 里的信息都拿去匹配
                    v = medicals_dict1[medical]
                    v.append(i[0])
                    medicals_dict1.update({medical:v})

    for k,v in medicals_dict1.items():   # 标准配伍集里的所有药物
        res = []
        for val in v:
            zhi = val.split(',')
            # print(zhi)
            u = medicals_dict4[k]
            u.append(zhi)
            medicals_dict4.update({k:u})
            for kk in zhi:
                res.append(kk)
        # v = medicals_dict1[k]
        v = res
        medicals_dict1.update({k:v})

    # for k,v in medicals_dict1:   # 标准配伍集里的所有药物
    #     print(k,len(v))
    result_data = []
    with open('func_5w_addresult.csv','r') as fr:
        reader = csv.reader(fr)
        count = 0
        funcs = []
        for i in reader:
            # print(i[2])
            # print(len(i[2].split(',')))
            if len(i[2].split(',')) > 2:    # 取药数量大于1的数据集
                result_data.append(i)
                for medical in medicals_dict2.keys():
                    if medical in i[4]  :     # 将 功效和主治 里的信息都拿去匹配 (功效在这个方子里,)

                        for hhh in medicals_dict1[medical]:    # 看 标注配伍集 里药物是否在此方剂中
                            if hhh in i[2]:
                                v = medicals_dict2[medical]
                                v.append(i[2])
                                medicals_dict2.update({medical:v})
                                break
    tongji = 0
    shuliangsort = []
    for k,v in medicals_dict2.items():
        if len(v)>200:
            # print(k,len(v),len(medicals_dict1[k]))
            tongji +=1
            resss =[k,len(v)]
            shuliangsort.append(resss)

    # print(tongji)
    shuliangsort = dict(shuliangsort)
    shuliangsort = sorted(shuliangsort.items(),key=operator.itemgetter(1),reverse=True)

    # for k,v in shuliangsort:
    #     print(k,v)

    for m1 in medicals_dict2.keys():       # 添加 负例
        length = 1.3*len(medicals_dict2[m1])
        random.shuffle(result_data)       # 随机打乱 数据集
        for i in result_data:
            if len(medicals_dict3[m1]) < length:
                if m1 not in i[4] and m1 not in i[5]:   # 将 功效和主治 里的信息都拿去匹配 (功效不在这个方子里)
                    v = medicals_dict3[m1]
                    v.append(i[2])
                    medicals_dict3.update({m1: v})
            else:
                break

    for folder in medicals_dict2.keys():   # 在 all 文件夹下建立各种功效的文件夹

        if len(medicals_dict2[folder])<200 or len(medicals_dict1[folder])<5:
            continue
        else:
            print(folder,len(medicals_dict1[folder]),len(medicals_dict2[folder]),len(medicals_dict4[folder]))
            position = 'fangji/'+folder
            if os.path.exists(position):
                pass
            else:
                os.makedirs(position)

            name1 = 'fangji/' + folder + '/' +folder+'_pres.csv'
            name2 = 'fangji/' + folder + '/' +'FuncFeature_'+folder + '.csv'
            name3 = 'fangji/' + folder + '/' + folder + '_evaluate.csv'
            name4 = 'fangji/' + folder +'/'+'Apriori_'+ folder +'_data.csv'
            with open(name4,'w') as fw4:
                writer4 = csv.writer(fw4)
                with open(name1,'w') as fw1:
                    writer1 = csv.writer(fw1)
                    with open(name2,'w') as fw2:
                        writer2 = csv.writer(fw2)
                        with open(name3, 'w') as fw3:
                            writer3 = csv.writer(fw3)
                            for v in medicals_dict2[folder]:

                                ss = v.split(',')
                                writer1.writerow(ss)
                                writer2.writerow([1])
                                res_m = []
                                for mm in range(0,len(ss),2):
                                    res_m.append(ss[mm])
                                writer4.writerow(res_m)
                            for u in medicals_dict3[folder]:

                                ss = u.split(',')
                                writer1.writerow(ss)
                                writer2.writerow([0])
                            for w in medicals_dict4[folder]:
                                writer3.writerow(w)

def zuihou():
    BM_func = {}
    with open('data/功效别名.csv', 'r') as fr:
        reader = csv.reader(fr)
        for hhh in reader:
            ss = hhh[1].strip().split(' ')
            BM_func.update({hhh[0]: ss})

    w = xlrd.open_workbook('data/最后筛选1.xls')
    sheet = w.sheets()[0]
    rows = sheet.nrows
    all = []
    for i in range(rows):
        name = sheet.cell_value(i,0).strip()
        num = sheet.cell_value(i,1)
        print(name,num)
        for k,v in BM_func.items():
            for val in v:
                if val==name:
                    name = k
                    break
        if len(name)>0:
            all.append([name,num])

    w = xlwt.Workbook(encoding='utf8')
    ws = w.add_sheet('name')
    haha = []
    count = 0
    for i in range(len(all)):
        name =all[i][0]
        if name not in haha and len(name)<4:
            ws.write(count, 0, name)
            ws.write(count, 1, all[i][1])
            haha.append(name)
            count +=1
    w.save('data/最后筛选2.xls')

def buque_jiliang():

    result_data = {}
    with open('func_5w_addresult.csv', 'r') as fr:
        reader = csv.reader(fr)
        count = 0
        funcs = []
        for i in reader:
            # print(i[2])       # 将 功效和主治 里的信息都拿去匹配 (功效在这个方子里)
            ss = i[2].split(',')
            for j in range(0,len(ss),2):
                if ss[j+1]!='None' and ss[j+1]!='0.0g':
                    if ss[j] not  in result_data.keys():
                        res = [ss[j+1]]
                        result_data.update({ss[j]:res})
                    else:
                        rrr = result_data[ss[j]]

                        rrr.append(ss[j+1])
                        result_data.update({ss[j]:rrr})

    with open('data/补缺剂量.csv','w') as fw:
        writer = csv.writer(fw)
        new = []
        for k,v in result_data.items():

                new.append([k,v])
        for jj in new:
            writer.writerow(jj)

def buque_jiliang_convert_g():
    dw_conver = {'两': 31.25, '钱': 3.125, '分': 0.3125, '厘': 0.03125, '毫': 0.003125, 'ml': 1, '毫克': 1, '毫升': 1,
                 'mg': 0.001,'钱匕':2.0,
                 '斤': 500.0, '铢': 0.5789, '克': 1.0, 'g': 1.0, 'kg': 1000.0, '千克': 1000.0, '撮': 10.35, '斗': 2000,
                 '公斤': 1000, '圭': 0.5, '合': 103.5, '斛': 51750, '钧': 6600, '累': 0.057, '升': 500, '石': 103500, '市斤': 500,
                 '市升': 1000, '黍': 0.0083, '桶': 20000.0, '籥': 10.0, '龠': 10.0,
                 '十两': 312.5, '八两': 249.0, '九两': 280.25, '七两': 218.75, '六两': 187.5, '五两': 156.25, '四两': 125.0,
                 '三两': 93.75, '二两': 62.5, '一两': 31.25, '半两': 15.625,
                 '一钱': 3.125, '三钱': 9.375, '一两半': 46.875, '二两半': 78.125,
                 '半钱': 1.5625, '半分': 0.15625, '半斤': 250.0
                 }
    with open('data/补缺剂量.csv','r') as fr:
        reader = csv.reader(fr)

        re330 = re.compile(r'(\d+)(.*)')
        re332 = re.compile(r'(\d+)\.(\d+)(.*)')
        all =[]
        re30 = re.compile(r'(\d+)(-|～|-|——|~|至)(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')
        re31 = re.compile(r'(.*)(，|,|:|;| |。|：)(\d+)(-|～|-|——|~)(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')
        # 剂量-小数
        re32 = re.compile(r'(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')  # min
        re33 = re.compile(r'(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')  # max
        re34 = re.compile(r'(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')  # min-max

        re35 = re.compile(r'(.*)(，|,|:|;| |。|：)(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')
        re36 = re.compile(r'(.*)(，|,|:|;| |。|：)(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')
        re37 = re.compile(
            r'(.*)(，|,|:|;| |。|：)(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')

        re38 = re.compile(r'(.*?)(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')
        for i in reader:
            danwei = i[1].replace(',','')
            danwei = danwei.replace("'",'')
            danwei = danwei.replace('[','')
            danwei = danwei.replace(']','')
            danwei = danwei.split(' ')

            TT = []
            TT.append(i[0])
            for dd in danwei:
                if re332.match(dd):     # 小数
                    result3 = re332.match(dd).groups()
                    # print(result3)
                    if result3[2] in dw_conver.keys():
                        rrr = float(result3[0]+'.'+result3[1])*dw_conver[result3[2]]
                        if rrr!=0.0:
                            TT.append(rrr)
                        # print(rrr)
                    else:
                        pass
                        # TT.append(dd)

                elif re330.match(dd):    # 整数
                    result = re330.match(dd.strip()).groups()
                    number = float(result[0].strip())
                    # print(i[1])
                    # print(result3)
                    if result[1] in dw_conver.keys():

                        rrr = float(result[0])*dw_conver[result[1]]
                        if rrr!=0.0:
                            TT.append(rrr)
                    elif result[1]=='两半':
                        TT.append(number*31.25+15.625)
                    elif result[1]=='两1钱半':
                        TT.append(number*31.25+3.125+3.125)
                    elif result[1]=='两2钱半':
                        TT.append(number*31.25+3.125*2+3.125)
                    elif result[1]=='两3钱半':
                        TT.append(number*31.25+3.125*3+3.125)
                    elif result[1]=='两4钱半':
                        TT.append(number*31.25+3.125*4+3.125)
                    elif result[1]=='两5钱半':
                        TT.append(number*31.25+3.125*5+3.125)
                    elif result[1]=='两6钱半':
                        TT.append(number*31.25+3.125*6+3.125)
                    elif result[1]=='两7钱半':
                        TT.append(number*31.25+3.125*7+3.125)
                    elif result[1]=='两8钱半':
                        TT.append(number*31.25+3.125*8+3.125)
                    elif result[1]=='两9钱半':
                        TT.append(number*31.25+3.125*9+3.125)
                    elif result[1]=='两1钱':
                        TT.append(number*31.25+3.125)
                    elif result[1]=='两2钱':
                        TT.append(number*31.25+3.125*2)
                    elif result[1]=='两3钱':
                        TT.append(number*31.25+3.125*3)
                    elif result[1]=='两4钱':
                        TT.append(number*31.25+3.125*4)
                    elif result[1]=='两5钱':
                        TT.append(number*31.25+15.625)
                    elif result[1]=='两6钱':
                        TT.append(number*31.25+3.125*6)
                    elif result[1]=='两7钱':
                        TT.append(number*31.25+3.125*7)
                    elif result[1]=='两8钱':
                        TT.append(number*31.25+3.125*8)
                    elif result[1]=='两9钱':
                        TT.append(number*31.25+3.125*9)
                    elif result[1] == '钱半':
                        TT.append(number * 3.125 + 1.5625)
                    elif result[1] == '斤半':
                        TT.append(number * 500.0 + 250)
                    elif result[1]=='钱1分':
                        TT.append(number*3.125+0.3125)
                    elif result[1]=='钱2分':
                        TT.append(number*3.125+0.3125*2)
                    elif result[1]=='钱3分':
                        TT.append(number*3.125+0.3125*3)
                    elif result[1]=='钱4分':
                        TT.append(number*3.125+1.25)
                    elif result[1]=='钱5分':
                        TT.append(number*3.125+1.5625)
                    elif result[1]=='钱6分':
                        TT.append(number*3.125+0.3125*6)
                    elif result[1]=='钱7分':
                        TT.append(number*3.125+2.1875)
                    elif result[1]=='钱8分':
                        TT.append(number*3.125+0.3125*8)
                    elif result[1]=='钱9分':
                        TT.append(number*3.125+0.3125*9)
                    elif result[1]=='两1分':
                        TT.append(number*31.25+0.3125)
                    elif result[1]=='两2分':
                        TT.append(number*31.25+0.3125*2)
                    elif result[1]=='两3分':
                        TT.append(number*31.25+0.3125*3)
                    elif result[1]=='两4分':
                        TT.append(number*31.25+0.3125*4)
                    elif result[1]=='两5分':
                        TT.append(number*31.25+0.3125*5)
                    elif result[1]=='两6分':
                        TT.append(number*31.25+0.3125*6)
                    elif result[1]=='两7分':
                        TT.append(number*31.25+0.3125*7)
                    elif result[1]=='两8分':
                        TT.append(number*31.25+0.3125*8)
                    elif result[1]=='两9分':
                        TT.append(number*31.25+0.3125*9)
                    elif result[1]=='分1厘':
                        TT.append(number*0.3125+0.03125)
                    elif result[1]=='分2厘':
                        TT.append(number*0.3125+0.03125*2)
                    elif result[1]=='分3厘':
                        TT.append(number*0.3125+0.03125*3)
                    elif result[1]=='分4厘':
                        TT.append(number*0.3125+0.03125*4)
                    elif result[1]=='分5厘':
                        TT.append(number*0.3125+0.03125*5)
                    elif result[1]=='分6厘':
                        TT.append(number*0.3125+0.03125*6)
                    elif result[1]=='分7厘':
                        TT.append(number*0.3125+0.03125*7)
                    elif result[1]=='分8厘':
                        TT.append(number*0.3125+0.03125*8)
                    elif result[1]=='分9厘':
                        TT.append(number*0.3125+0.03125*9)
                    elif result[1]=='两1铢':
                        TT.append(number*31.25+0.5789)
                    elif result[1]=='两2铢':
                        TT.append(number*31.25+0.5789*2)
                    elif result[1]=='两3铢':
                        TT.append(number*31.25+0.5789*3)
                    elif result[1]=='两4铢':
                        TT.append(number*31.25+0.5789*4)
                    elif result[1]=='两5铢':
                        TT.append(number*31.25+0.5789*5)
                    elif result[1]=='两6铢' or result[1]=='两6株':
                        TT.append(number*31.25+0.5789*6)
                    elif result[1]=='两7铢':
                        TT.append(number*31.25+0.5789*7)
                    elif result[1]=='两8铢':
                        TT.append(number*31.25+0.5789*8)
                    elif result[1]=='两9铢':
                        TT.append(number*31.25+0.5789*9)
                    elif result[1]=='两18铢':
                        TT.append(number*31.25+0.5789*18)
                    elif result[1]=='斤1两':
                        TT.append(number*500+31.25*1)
                    elif result[1]=='斤2两':
                        TT.append(number*500+31.25*2)
                    elif result[1]=='斤3两':
                        TT.append(number*500+31.25*3)
                    elif result[1]=='斤4两':
                        TT.append(number*500+31.25*4)
                    elif result[1]=='斤5两':
                        TT.append(number*500+31.25*5)
                    elif result[1]=='斤6两':
                        TT.append(number*500+31.25*6)
                    elif result[1]=='斤7两':
                        TT.append(number*500+31.25*7)
                    elif result[1]=='斤8两':
                        TT.append(number*500+31.25*8)
                    elif result[1]=='斤9两':
                        TT.append(number*500+31.25*9)
                    elif result[1]=='斤12两':
                        TT.append(number*500+31.25*12)

                    elif re30.match(dd):
                        result3 = re30.match(dd).groups()
                        # print(result3)
                        if result3[3] in dw_conver.keys():
                            rrr = (float(result3[0])+float(result3[2]))/2 * dw_conver[result3[3]]
                            if rrr != 0.0:
                                TT.append(rrr)
                        else:
                            pass
                            # TT.append(dd)
                            # print(dd)
                    else:
                        # all.append(TT)
                        pass
                        # TT.append(dd)
                        # print(dd)
                else:
                    if rrr != 0.0:
                        TT.append(rrr)
                    # print(rrr)

            all.append(TT)

    with open('data/补缺剂量1.csv','w') as fw:
        writer =csv.writer(fw)

        cc = 0
        xiaoyu = 0
        for i in all:
            new_data = [i[0]]
            jiliang = i[1:]
            # print(jiliang)
            jiliang = sorted(jiliang)
            # print(jiliang)

            if len(jiliang)==0:
                # print(new_data)
                cc +=1
            else:
                D = 0.0

                if len(jiliang)==1:              #  中位数
                    Med = jiliang[0]
                    E = sum(jiliang) / len(jiliang)   # 方差
                    for zhi in range(len(jiliang)):
                        D += (E - jiliang[zhi]) ** 2
                    D = (D / len(jiliang))**(1/2)
                elif len(jiliang)==2:                # 只有2个，用 期望
                    Med = (jiliang[0]+jiliang[1]) / 2
                    E = sum(jiliang) / len(jiliang)     # 方差
                    for zhi in range(len(jiliang)):
                        D += (E - jiliang[zhi]) ** 2
                    D = (D / len(jiliang))**(1/2)
                elif len(jiliang)==3:
                    # length = int(len(jiliang)/2)    #  中位数
                    # length = 1
                    # Med = jiliang[length]
                    Med = (jiliang[0]+jiliang[1]+jiliang[2]) / 3
                    E = sum(jiliang) / len(jiliang)
                    for zhi in range(len(jiliang)):
                        D += (E - jiliang[zhi]) ** 2
                    D = (D / len(jiliang))**(1/2)
                else:

                    Med = sum(jiliang) / len(jiliang)
                    E = sum(jiliang) / len(jiliang)
                    for zhi in range(len(jiliang)):
                        D += (E - jiliang[zhi]) ** 2
                    D = (D / len(jiliang)) ** (1 / 2)

                new_data.append(Med)

                Min = Med - D/((len(jiliang))**(1/2))*3
                Max = Med + D/((len(jiliang))**(1/2))*3

                if Min < 0:
                    print(jiliang)
                    print(Min)
                    Min = jiliang[0]
                    xiaoyu +=1

                new_data.append(Min)
                new_data.append(Max)

                writer.writerow(new_data)
        print('total:',len(all))
        print('为空:',cc)
        print('小于0:',xiaoyu)



if __name__=='__main__':

    # buque_jiliang()
    # buque_jiliang_convert_g()
    # zuihou()    # 清理别名 最后筛选
    # word2vec()    # 预训练词向量
    # bieming()        # 标准配伍别名
    # quchong1()

    # doit()     # 清洗方剂功效
    # add_fangjis()
    process_medicals()   # 清洗 方剂药组

    huafen()    # 按照各个功效划分数据,并存储
    #









