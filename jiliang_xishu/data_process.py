# -*-coding:utf-8 -*-
__author__ = '$'

import csv
import pandas as pd
import re
import numpy as np
import operator
import os

def k_top(apriori_result):

    k_data = []
    max_data = []
    for k in range(1,6):
        max_medical = []
        max = 0.0
        for ak, av in apriori_result.items():
            if len(ak)==k and av > max:
                max = av
                res =[]
                for i in ak:
                    res.append(i.strip())
                max_medical = res
        k_data.append(max_medical)
        max_data.append(max)
    return k_data,max_data

def count_biaozhuncha(v,u):
    sum_result = 0
    for i in v:
        sum_result += (float(i)-u)**2
    return (sum_result/len(v))**0.5


def process1():

    fw = open('data/data1.csv','w')
    writer = csv.writer(fw)
    writer.writerow(['药名','别名','功效','min','max','单位'])

    df = pd.read_excel('data/中药.xlsx',na_values='')
    print(df.columns)
    data = df.loc[:,['药材名称','别名','功能主治','用法用量']].values
    # 剂量-整数
    re30 =  re.compile(r'(\d+)(-|～|-|——|~|至)(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')
    re31 = re.compile(r'(.*)(，|,|:|;| |。|：)(\d+)(-|～|-|——|~)(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')
    # 剂量-小数
    re32 = re.compile(r'(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')   # min
    re33 = re.compile(r'(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')   # max
    re34 = re.compile(r'(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')  # min-max

    re35 = re.compile(r'(.*)(，|,|:|;| |。|：)(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')
    re36 = re.compile(r'(.*)(，|,|:|;| |。|：)(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')
    re37 = re.compile(r'(.*)(，|,|:|;| |。|：)(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')

    re38 = re.compile(r'(.*?)(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')

    for  i in data:

        if re30.match(str(i[3])):
            result3 = re30.match(str(i[3])).groups()
            # print(i[3])
            # print(result3)

            writer.writerow([i[0],i[1],i[2],result3[0],result3[2],result3[3]])
        elif re31.match(str(i[3])):
            result3 = re31.match(str(i[3])).groups()

            # print(i[3])
            # print(result3)
            writer.writerow([i[0], i[1], i[2], result3[2],result3[4],result3[5]])

        elif re32.match(str(i[3])):
            result3 = re32.match(str(i[3])).groups()

            # print(i[3])
            # print(result3)
            min = result3[0]+'.'+result3[1]
            # print(min)

            writer.writerow([i[0], i[1], i[2], float(min), result3[3], result3[4]])
        elif re33.match(str(i[3])):
            result3 = re33.match(str(i[3])).groups()

            # print(i[3])
            # print(result3)
            max = result3[2]+'.'+result3[3]
            # print(float(max))
            writer.writerow([i[0], i[1], i[2], result3[0], float(max), result3[4]])

        elif re34.match(str(i[3])):
            result3 = re34.match(str(i[3])).groups()

            # print(i[3])
            # print(result3)
            min = result3[0]+'.'+result3[1]
            max = result3[3]+'.'+result3[4]

            writer.writerow([i[0], i[1], i[2],float(min), float(max), result3[5]])

        elif re35.match(str(i[3])):
            result3 = re35.match(str(i[3])).groups()
            min = result3[2]+'.'+result3[3]
            # print(i[3])
            # print(result3)
            writer.writerow([i[0], i[1], i[2], float(min), result3[5], result3[6]])

        elif re36.match(str(i[3])):
            result3 = re36.match(str(i[3])).groups()
            max = result3[4]+'.'+result3[5]
            # print(i[3])
            # print(result3)
            writer.writerow([i[0], i[1], i[2], result3[2], float(max), result3[6]])
        elif re37.match(str(i[3])):
            result3 = re37.match(str(i[3])).groups()
            min = result3[2]+'.'+result3[3]
            max = result3[5]+'.'+result3[6]
            # print(i[3])
            # print(result3)
            writer.writerow([i[0], i[1], i[2], float(min), float(max), result3[7]])

        elif re38.match(str(i[3])):
            result3 = re38.match(str(i[3])).groups()
            # print(i[3])
            # print(result3)
            writer.writerow([i[0], i[1], i[2], '', '', i[3]])
        else:

            print(i[3])
            # print(result3)
            writer.writerow([i[0], i[1], i[2], '','',i[3]])
            pass


def process2():

    fw = open('data/data2.csv', 'w')
    writer = csv.writer(fw)
    writer.writerow(['药名', 'min', 'max', '单位'])

    df = pd.read_excel('data/中国药典.xls', na_values='',header=1)
    print(df.columns)

    data = df.loc[:,['品名','用法用量修订后 ']].values

    re30 = re.compile(r'(\d+)(-|～|-|——|~|至)(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')
    re32 = re.compile(r'(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')
    re33 = re.compile(r'(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')
    re34 = re.compile(r'(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)\.(\d+)(g|两|钱|只|滴|个|分|枚|厘|片|朵|条|根|ml| |mg)(.*)')
    for i in data:

        if re30.match(str(i[1]).strip()):
            result3 = re30.match(str(i[1]).strip()).groups()
            # print(i[1])
            # print(result3)
            writer.writerow([i[0].strip(),result3[0],result3[2],result3[3]])

        elif re32.match(str(i[1]).strip()):
            result3 = re32.match(str(i[1]).strip()).groups()
            # print(i[1])
            # print(result3)

            min = result3[0]+'.'+result3[1]
            writer.writerow([i[0].strip(), float(min), result3[3], result3[4]])
        elif re33.match(str(i[1]).strip()):
            result3 = re33.match(str(i[1]).strip()).groups()
            # print(i[1])
            # print(result3)
            max = result3[2]+'.'+result3[3]
            writer.writerow([i[0].strip(), result3[0], float(max), result3[4]])

        elif re34.match(str(i[1]).strip()):
            result3 = re34.match(str(i[1])).groups()
            # print(i[1])
            # print(result3)

            min = result3[0]+'.'+result3[1]
            max = result3[3]+'.'+result3[4]
            writer.writerow([i[0].strip(), float(min),float(max), result3[5]])

        else: # 手动整理剩下的
            print(i[1])
            writer.writerow([i[0].strip(),'','',i[1]])



def process3(preName):

    danwei0 = re.compile(r'^(\d+)\.(\d+)(.*)')
    danwei1 = re.compile(r'^(\d+)(.*)')
    danwei2 = re.compile(r'^(.*)')
    dd = set()

    medical = {}

    path1 = 'data/fangji/'+preName+'/'+preName+'_pres.csv'
    fr = open(path1,'r')
    reader = csv.reader(fr)

    for i in reader:
        # print(i)
        for j in range(0,len(i),2):
            # print(i[j+1])
            # print(i[j])
            result_res = ' '
            if danwei0.match(str(i[j+1]).strip()):
                result = danwei0.match(str(i[j+1])).groups()
                dd.add(result[2])
                # print(result)
                result_res = result[2]
            elif danwei1.match(str(i[j+1]).strip()):
                result = danwei1.match(str(i[j+1])).groups()
                dd.add(result[1])
                # print(result)
                result_res = result[1]

            elif danwei2.match(str(i[j+1]).strip()):
                result = danwei2.match(str(i[j+1])).groups()
                # print(result)
                dd.add(result[0])
                result_res = result[0]

            else:
                pass

            if i[j].strip() != '':
                if i[j].strip() not in medical.keys() :
                    # print(result_res)
                    medical.update({i[j].strip():[result_res]})
                else:
                    res = medical[i[j].strip()]
                    # print(res)
                    if result_res not in res:
                        res.append(result_res)

    path2 = 'data/fangji/'+preName+'/' + preName + '_danwei.txt'
    with open(path2,'w') as fw:
        fw.write(' '.join(dd))

    # print(dd)        # 所有单位
    # print('单位数：',len(dd))
    # print(medical)
    # print('药物数：',len(medical.keys()))

    path3= 'data/fangji/'+ preName+'/'+preName + '_danwei.csv'

    with open(path3,'w') as fw:
        writer = csv.writer(fw)
        for k,v in medical.items():
            writer.writerow([k,','.join(v)])


def process4(preName,dw_conver):
    buque =[]
    with open('data/补缺剂量.csv','r') as fr:
        reader = csv.reader(fr)
        for i in reader:
            buque.append(i)

    fr_ji_1 = open('data/data1.csv','r')
    reader_ji_1 = csv.reader(fr_ji_1)
    jiliang_1 = {}
    for i in reader_ji_1:
        # print(i)
        if i[5].strip()=='g':
            jiliang_1.update({i[0].strip():[i[3],i[4]]})
        elif i[5].strip()=='两' and i[0].strip() not in jiliang_1.keys():
            jiliang_1.update({i[0].strip():[float(i[3])*31.25,float(i[4])*31.25]})
        elif i[5].strip()=='钱' and i[0].strip() not in jiliang_1.keys():
            jiliang_1.update({i[0].strip():[float(i[3])*3.125,float(i[4])*3.125]})
        elif i[5].strip()=='分' and i[0].strip() not in jiliang_1.keys():
            jiliang_1.update({i[0].strip():[float(i[3])*0.3125,float(i[4])*0.3125]})
        else:
            pass
            # print(i)
    fr_ji = open('data/药典剂量范围.csv','r')
    reader_ji = csv.reader(fr_ji)

    jiliang = {}
    for i in reader_ji:
        # print(i)
        if i[1].strip()!='' and i[2].strip()!='':
            jiliang.update({i[0].strip():[i[1],i[2]]})
    # print(jiliang)

    danwei0 = re.compile(r'^(\d+)\.(\d+)(.+)')
    danwei1 = re.compile(r'^(\d+)(.*)')
    re30 = re.compile(
        r'(\d+)(-|～|-|——|~|至)(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')
    re31 = re.compile(
        r'(.*)(，|,|:|;| |。|：)(\d+)(-|～|-|——|~)(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')
    # 剂量-小数
    re32 = re.compile(
        r'(\d+)\.(\d+)(-|～|-|——|~|至)(\d+)(g|kg|ml|mg|l|个|籥|圭|累|钧|斛|钱半|钱匕|茶匙|钱|市升|片|根|条|份|滴|张|枚|具|朵|只|粒|茎|两半|两|斤半|斤|公斤|挺|对|头|L|ML|分|节|cm|握|株|毫克|克|千克|升|毫升|余寸|寸|合|把|块|颗|铢|勺|团|碗|厘|字|双|页|石|束|盏|撮|贴|面|尺|截|匙|瓢|支|斗|文|杯|帖|叶|锭|角|方|全付|付|重|梃|尾|副|秤|丸|箔)(.*)')  # min

    path1 = 'data/fangji/' + preName+'/'+preName + '_last_data.csv'
    fw = open(path1,'w')
    writer = csv.writer(fw)

    path2 = 'data/fangji/' + preName +'/'+preName+ '_pres.csv'
    with open(path2,'r') as fr:
        reader = csv.reader(fr)
        for i in reader:
            # print(i)
            medical = []
            for j in range(0,len(i),2):
                # print(i[j])
                # print(i[j+1])
                if danwei0.match(str(i[j + 1])):       # 小数
                    result = danwei0.match(str(i[j + 1])).groups()
                    # print(result)
                    number = float(result[0].strip() + '.' + result[1].strip())
                    if result[2].strip() in dw_conver.keys():
                        # print(number)
                        # print(dw_conver[result[2]])
                        medical.append(i[j])
                        medical.append(number*dw_conver[result[2].strip()])
                    else:
                        # print(i[j],i[j+1])
                        cccc = 0
                        for bbbb in buque:
                            if bbbb[0].strip() == i[j].strip():
                                # print('补缺')
                                medical.append(i[j].strip())
                                medical.append(bbbb[1])
                                break
                            cccc += 1
                        if cccc == len(buque):      # 补缺里的不够  (只有2个药物)
                            if i[j].strip() in jiliang.keys():  # 药典——剂量
                                # print(jiliang[i[j]])
                                res = (float(jiliang[i[j]][0]) + float(jiliang[i[j]][1])) / 2
                                # print(res)
                                medical.append(i[j])
                                medical.append(res)

                            elif i[j].strip() in jiliang_1.keys():  # 替补剂量
                                # print(i[j])
                                res = (float(jiliang_1[i[j]][0]) + float(jiliang_1[i[j]][1])) / 2
                                # print(res)
                                medical.append(i[j].strip())
                                medical.append(res)
                            else:
                                # print(i[j], i[j + 1])
                                medical.append(i[j].strip())
                                medical.append(1.0)
                            # print(i[j], i[j + 1])

                elif danwei1.match(str(i[j + 1]).strip()):   # 整数
                    result = danwei1.match(str(i[j + 1])).groups()
                    number = float(result[0].strip())
                    # print(result)
                    if result[1].strip() in dw_conver.keys():
                        medical.append(i[j])
                        # print(number)
                        # print(number*dw_conver[result[1].strip()])
                        medical.append(number*dw_conver[result[1].strip()])
                    elif result[1]=='两半':
                        medical.append(i[j])
                        medical.append(number*31.25+15.625)
                    elif result[1]=='两1钱半':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125+3.125)
                    elif result[1]=='两2钱半':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*2+3.125)
                    elif result[1]=='两3钱半':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*3+3.125)
                    elif result[1]=='两4钱半':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*4+3.125)
                    elif result[1]=='两5钱半':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*5+3.125)
                    elif result[1]=='两6钱半':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*6+3.125)
                    elif result[1]=='两7钱半':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*7+3.125)
                    elif result[1]=='两8钱半':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*8+3.125)
                    elif result[1]=='两9钱半':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*9+3.125)
                    elif result[1]=='两1钱':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125)
                    elif result[1]=='两2钱':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*2)
                    elif result[1]=='两3钱':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*3)
                    elif result[1]=='两4钱':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*4)
                    elif result[1]=='两5钱':
                        medical.append(i[j])
                        medical.append(number*31.25+15.625)
                    elif result[1]=='两6钱':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*6)
                    elif result[1]=='两7钱':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*7)
                    elif result[1]=='两8钱':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*8)
                    elif result[1]=='两9钱':
                        medical.append(i[j])
                        medical.append(number*31.25+3.125*9)
                    elif result[1] == '钱半':
                        medical.append(i[j])
                        medical.append(number * 3.125 + 1.5625)
                    elif result[1] == '斤半':
                        medical.append(i[j])
                        medical.append(number * 500.0 + 250)
                    elif result[1]=='钱1分':
                        medical.append(i[j])
                        medical.append(number*3.125+0.3125)
                    elif result[1]=='钱2分':
                        medical.append(i[j])
                        medical.append(number*3.125+0.3125*2)
                    elif result[1]=='钱3分':
                        medical.append(i[j])
                        medical.append(number*3.125+0.3125*3)
                    elif result[1]=='钱4分':
                        medical.append(i[j])
                        medical.append(number*3.125+1.25)
                    elif result[1]=='钱5分':
                        medical.append(i[j])
                        medical.append(number*3.125+1.5625)
                    elif result[1]=='钱6分':
                        medical.append(i[j])
                        medical.append(number*3.125+0.3125*6)
                    elif result[1]=='钱7分':
                        medical.append(i[j])
                        medical.append(number*3.125+2.1875)
                    elif result[1]=='钱8分':
                        medical.append(i[j])
                        medical.append(number*3.125+0.3125*8)
                    elif result[1]=='钱9分':
                        medical.append(i[j])
                        medical.append(number*3.125+0.3125*9)
                    elif result[1]=='两1分':
                        medical.append(i[j])
                        medical.append(number*31.25+0.3125)
                    elif result[1]=='两2分':
                        medical.append(i[j])
                        medical.append(number*31.25+0.3125*2)
                    elif result[1]=='两3分':
                        medical.append(i[j])
                        medical.append(number*31.25+0.3125*3)
                    elif result[1]=='两4分':
                        medical.append(i[j])
                        medical.append(number*31.25+0.3125*4)
                    elif result[1]=='两5分':
                        medical.append(i[j])
                        medical.append(number*31.25+0.3125*5)
                    elif result[1]=='两6分':
                        medical.append(i[j])
                        medical.append(number*31.25+0.3125*6)
                    elif result[1]=='两7分':
                        medical.append(i[j])
                        medical.append(number*31.25+0.3125*7)
                    elif result[1]=='两8分':
                        medical.append(i[j])
                        medical.append(number*31.25+0.3125*8)
                    elif result[1]=='两9分':
                        medical.append(i[j])
                        medical.append(number*31.25+0.3125*9)

                    elif result[1]=='分1厘':
                        medical.append(i[j])
                        medical.append(number*0.3125+0.03125)
                    elif result[1]=='分2厘':
                        medical.append(i[j])
                        medical.append(number*0.3125+0.03125*2)
                    elif result[1]=='分3厘':
                        medical.append(i[j])
                        medical.append(number*0.3125+0.03125*3)
                    elif result[1]=='分4厘':
                        medical.append(i[j])
                        medical.append(number*0.3125+0.03125*4)
                    elif result[1]=='分5厘':
                        medical.append(i[j])
                        medical.append(number*0.3125+0.03125*5)
                    elif result[1]=='分6厘':
                        medical.append(i[j])
                        medical.append(number*0.3125+0.03125*6)
                    elif result[1]=='分7厘':
                        medical.append(i[j])
                        medical.append(number*0.3125+0.03125*7)
                    elif result[1]=='分8厘':
                        medical.append(i[j])
                        medical.append(number*0.3125+0.03125*8)
                    elif result[1]=='分9厘':
                        medical.append(i[j])
                        medical.append(number*0.3125+0.03125*9)
                    elif result[1]=='两1铢':
                        medical.append(i[j])
                        medical.append(number*31.25+0.5789)
                    elif result[1]=='两2铢':
                        medical.append(i[j])
                        medical.append(number*31.25+0.5789*2)
                    elif result[1]=='两3铢':
                        medical.append(i[j])
                        medical.append(number*31.25+0.5789*3)
                    elif result[1]=='两4铢':
                        medical.append(i[j])
                        medical.append(number*31.25+0.5789*4)
                    elif result[1]=='两5铢':
                        medical.append(i[j])
                        medical.append(number*31.25+0.5789*5)
                    elif result[1]=='两6铢' or result[1]=='两6株':
                        medical.append(i[j])
                        medical.append(number*31.25+0.5789*6)
                    elif result[1]=='两7铢':
                        medical.append(i[j])
                        medical.append(number*31.25+0.5789*7)
                    elif result[1]=='两8铢':
                        medical.append(i[j])
                        medical.append(number*31.25+0.5789*8)
                    elif result[1]=='两9铢':
                        medical.append(i[j])
                        medical.append(number*31.25+0.5789*9)
                    elif result[1]=='两18铢':
                        medical.append(i[j])
                        medical.append(number*31.25+0.5789*18)

                    elif result[1]=='斤1两':
                        medical.append(i[j])
                        medical.append(number*500+31.25*1)
                    elif result[1]=='斤2两':
                        medical.append(i[j])
                        medical.append(number*500+31.25*2)
                    elif result[1]=='斤3两':
                        medical.append(i[j])
                        medical.append(number*500+31.25*3)
                    elif result[1]=='斤4两':
                        medical.append(i[j])
                        medical.append(number*500+31.25*4)
                    elif result[1]=='斤5两':
                        medical.append(i[j])
                        medical.append(number*500+31.25*5)
                    elif result[1]=='斤6两':
                        medical.append(i[j])
                        medical.append(number*500+31.25*6)
                    elif result[1]=='斤7两':
                        medical.append(i[j])
                        medical.append(number*500+31.25*7)
                    elif result[1]=='斤8两':
                        medical.append(i[j])
                        medical.append(number*500+31.25*8)
                    elif result[1]=='斤9两':
                        medical.append(i[j])
                        medical.append(number*500+31.25*9)
                    elif result[1]=='斤12两':
                        medical.append(i[j])
                        medical.append(number*500+31.25*12)

                    else:
                        if re30.match(str(i[j + 1]).strip()):
                            result = re30.match(str(i[j + 1])).groups()
                            number = (float(result[0])+float(result[2]))/2
                            if result[3] in dw_conver:
                                number = number * dw_conver[result[3]]
                                medical.append(i[j])
                                medical.append(number)
                            else:    # 补缺
                                cccc = 0
                                for bbbb in buque:
                                    if bbbb[0].strip() == i[j].strip():
                                        # print('补缺')
                                        medical.append(i[j].strip())
                                        medical.append(bbbb[1])
                                        break
                                    cccc += 1
                                if cccc == len(buque):      # 补缺里的不够  (只有40个药物)
                                    if i[j].strip() in jiliang.keys():  # 药典——剂量
                                        # print(jiliang[i[j]])
                                        res = (float(jiliang[i[j]][0]) + float(jiliang[i[j]][1])) / 2
                                        # print(res)
                                        medical.append(i[j])
                                        medical.append(res)

                                    elif i[j].strip() in jiliang_1.keys():     # 替补剂量
                                            # print(i[j])
                                            res = (float(jiliang_1[i[j]][0]) + float(jiliang_1[i[j]][1])) / 2
                                            # print(res)
                                            medical.append(i[j].strip())
                                            medical.append(res)
                                    else:
                                        # print(i[j], i[j + 1])
                                        medical.append(i[j].strip())
                                        medical.append(1.0)

                        else:
                            cccc = 0
                            for bbbb in buque:
                                if bbbb[0].strip() == i[j].strip():
                                    # print('补缺')
                                    medical.append(i[j].strip())
                                    medical.append(bbbb[1])
                                    break
                                cccc += 1
                            if cccc == len(buque):  # 补缺里的不够  (只有40个药物)
                                if i[j].strip() in jiliang.keys():  # 药典——剂量
                                    # print(jiliang[i[j]])
                                    res = (float(jiliang[i[j]][0]) + float(jiliang[i[j]][1])) / 2
                                    # print(res)
                                    medical.append(i[j])
                                    medical.append(res)
                                elif i[j].strip() in jiliang_1.keys():  # 替补剂量
                                    # print(i[j])
                                    res = (float(jiliang_1[i[j]][0]) + float(jiliang_1[i[j]][1])) / 2
                                    # print(res)
                                    medical.append(i[j].strip())
                                    medical.append(res)
                                else:
                                    # print(i[j], i[j + 1])
                                    medical.append(i[j].strip())
                                    medical.append(1.0)

                else:

                    if i[j+1].strip() in dw_conver.keys():

                        medical.append(i[j])
                        medical.append(dw_conver[i[j+1].strip()])

                    elif i[j+1].strip()=='None':
                        cccc = 0
                        for bbbb in buque:
                            if bbbb[0].strip() == i[j].strip():
                                # print('补缺')
                                medical.append(i[j].strip())
                                medical.append(bbbb[1])
                                break
                            cccc += 1
                        if cccc == len(buque):     # 补缺里的不够  (只有40个药物)
                            if i[j].strip() in jiliang.keys():  # 药典——剂量
                                # print(jiliang[i[j]])
                                res = (float(jiliang[i[j]][0]) + float(jiliang[i[j]][1])) / 2
                                # print(res)
                                medical.append(i[j])
                                medical.append(res)
                            elif i[j].strip() in jiliang_1.keys():  # 替补剂量
                                # print(i[j])
                                res = (float(jiliang_1[i[j]][0]) + float(jiliang_1[i[j]][1])) / 2
                                # print(res)
                                medical.append(i[j].strip())
                                medical.append(res)
                            else:
                                # print(i[j], i[j + 1])
                                medical.append(i[j].strip())
                                medical.append(1.0)

                    else:   # 没有
                        print(i[j+1])
                        pass
            # print(medical)
            writer.writerow(medical)


def process5(Name):

    buque = []
    with open('data/补缺剂量.csv', 'r') as fr:
        reader = csv.reader(fr)
        for i in reader:
            buque.append(i)

    fr_ji_1 = open('data/data1.csv', 'r')
    reader_ji_1 = csv.reader(fr_ji_1)
    jiliang_1 = {}
    for i in reader_ji_1:
        # print(i)
        if i[5].strip() == 'g':
            jiliang_1.update({i[0].strip(): [i[3], i[4]]})
        elif i[5].strip() == '两' and i[0].strip() not in jiliang_1.keys():
            jiliang_1.update({i[0].strip(): [float(i[3]) * 31.25, float(i[4]) * 31.25]})
        elif i[5].strip() == '钱' and i[0].strip() not in jiliang_1.keys():
            jiliang_1.update({i[0].strip(): [float(i[3]) * 3.125, float(i[4]) * 3.125]})
        elif i[5].strip() == '分' and i[0].strip() not in jiliang_1.keys():
            jiliang_1.update({i[0].strip(): [float(i[3]) * 0.3125, float(i[4]) * 0.3125]})
        else:
            pass
    fr_ji_1.close()

    fr_ji = open('data/药典剂量范围.csv', 'r')
    reader_ji = csv.reader(fr_ji)
    jiliang = {}
    for i in reader_ji:
        # print(i)
        if i[1]!='' and i[2]!='':
            jiliang.update({i[0].strip(): [i[1], i[2]]})
    fr_ji.close()
    # print(jiliang)
    n = 10.0
    m = 50.0

    Tot = 0
    low = 0
    medd = 0
    hig = 0
    for preName in Name:

        # if preName !='清肝':
        #     continue

        print(preName)
        AVG_xd = {}           # 记录每一种功效下相对剂量（小数）

        path0 = 'data/fangji/'+preName+'/'+'FuncFeature_'+preName+'.csv'
        POS = 0
        POS_c = 0
        with open(path0,'r') as fr0:
            reader0 = csv.reader(fr0)
            for k in reader0:
                if k[0]=='1':
                    POS +=1
        print(POS)

        path1 = 'data/fangji/' +preName+'/'+ preName + '_train_data.txt'
        fw = open(path1,'w')
        path2 = 'data/fangji/' +preName+'/'+ preName + '_last_data.csv'
        with open(path2,'r') as fr:
            reader = csv.reader(fr)
            for i in reader:
                medical = []
                sum_row = 0.0
                for j in range(0,len(i),2):

                    # print(i[j],i[j+1])
                    ssss = i[j].split(' ')
                    i[j] = ''.join(ssss)
                    if '生' in i[j] and '姜' in i[j]:
                       i[j]= '生姜'
                    if '白花蛇' in i[j] and '舌草' in i[j]:
                       i[j]= '白花蛇舌草'
                    if '炙甘' in i[j] and '草' in i[j]:
                       i[j]= '炙甘草'
                    if '炙' in i[j] and '甘草' in i[j]:
                       i[j]= '炙甘草'
                    if '刺' in i[j] and '蒺藜' in i[j]:
                       i[j]= '刺蒺藜'
                    if '白茅' in i[j] and '根' in i[j]:
                       i[j]= '白茅根'
                    if '马' in i[j] and '齿苋' in i[j]:
                       i[j]= '马齿苋'

                    cccc = 0
                    x = float(i[j + 1])
                    for bbbb in buque:
                        if bbbb[0].strip() == i[j].strip():
                            # print('补缺')
                            Tot +=1
                            if float(bbbb[2])==float(bbbb[3]):
                                G = float(bbbb[2])
                                medical.append(i[j].strip())
                                sum_row +=G
                                medical.append(G)
                                break
                            if x < float(bbbb[2]):
                                G = n
                                low += 1
                            elif float(bbbb[2]) <= x <= float(bbbb[3]):
                                G = (x - float(bbbb[2])) / (float(bbbb[3]) - float(bbbb[2])) * m + n
                                medd += 1
                            else:
                                G = (m + n)
                                hig += 1

                            medical.append(i[j].strip())
                            sum_row += G
                            medical.append(G)
                            break
                        cccc += 1
                    if cccc != len(buque):
                        continue
                    else :  # 补缺里的不够
                        if i[j].strip() in jiliang.keys():
                            # print(jiliang[i[j]])
                            a = float(jiliang[i[j]][0])
                            b = float(jiliang[i[j]][1])
                            x = float(i[j+1])
                            Tot +=1
                            if x < a :
                                G = n
                                low += 1
                            elif a <= x <= b:
                                G = (x-a)/(b-a)*m+n
                                medd += 1
                            else:
                                G = (m+n)
                                hig += 1
                            medical.append(i[j].strip())
                            sum_row += G
                            medical.append(G)
                        elif i[j].strip() in jiliang_1.keys():
                            # print(jiliang_1[i[j]])
                            a = float(jiliang_1[i[j]][0])
                            b = float(jiliang_1[i[j]][1])
                            x = float(i[j+1])
                            Tot += 1
                            if x < a :
                                G = n
                                low +=1
                            elif a <= x <= b:
                                G = (x-a)/(b-a)*m+n
                                medd +=1
                            else:
                                G = (m+n)
                                hig +=1
                            medical.append(i[j].strip())
                            sum_row += G
                            medical.append(G)
                        else:         # 没在上面两个药典中
                            # print(i[j])
                            # print(i[j+1])
                            medical.append(i[j].strip())
                            sum_row += G
                            medical.append((m+n+n)/2)
                res_kkk = []

                max_v = max([medical[nn] for nn in range(1,len(medical),2)])
                for kkk in range(0,len(medical),2):
                    res_kkk.append(medical[kkk])
                    # res_kkk.append(float(medical[kkk+1])/sum_row)   # 分母为和
                    # res_kkk.append(float(medical[kkk+1]) / max_v)    #    分母为最大的那一个值
                    res_kkk.append(float(medical[kkk + 1]))     # 直接添加相对剂量，不做归一化

                if POS_c < POS:      # 只从正例里找相对剂量
                    for lin  in range(0,len(res_kkk),2):
                        if res_kkk[lin].strip() not in AVG_xd:
                            AVG_xd.update({res_kkk[lin].strip(): [res_kkk[lin+1]]})
                        else:
                            hhh = AVG_xd[res_kkk[lin].strip()]
                            hhh.append(res_kkk[lin+1])
                            AVG_xd.update({res_kkk[lin].strip():hhh})
                POS_c +=1
            # writer.writerow(medical)
                fw.write(','.join([str(k) for k in medical])+'\n')
        fw.close()
        path2_5 = 'data/fangji/' + preName + '/' + preName + '_avg_float.csv'
        with open(path2_5,'w') as fw2_5:
            writer2_5 = csv.writer(fw2_5)
            for k,v in AVG_xd.items():
                # print(k,v)
                writer2_5.writerow([k,sum(v)/len(v)])

        all_medical = set()   # 方剂集下所有的药物
        count_medical = {}
        path3 = 'data/fangji/'+preName+'/' + preName + '_train_data.txt'
        with open(path3,'r') as fr1:
            reader1 = csv.reader(fr1)
            nn = 0
            for i in reader1:
                # print(nn+1,i)
                nn += 1
                for j in range(0,len(i),2):
                    # print(i[j])
                    all_medical.add(i[j])
                    if i[j].strip() not in count_medical.keys():
                        count_medical.update({i[j].strip():1})
                    else:
                        res = count_medical[i[j].strip()]

                        count_medical.update({i[j].strip():res + 1})
            # print('nn:',nn)
        # print('方剂集下的所有药物：',len(all_medical))
        # print(count_medical)

        med = []
        path4 = 'data/fangji/'+preName+'/' + preName + '_count_medical.csv'
        dimension = 0
        with open(path4,'w') as fw1:
            writer1 = csv.writer(fw1)
            for k,v in sorted(count_medical.items(),key=operator.itemgetter(1),reverse=True):
                if float(v)>1:
                    writer1.writerow([k,v])
                    dimension +=1
                    if k not in med:
                        if k.strip()!='':
                            # print(k,v)
                            med.append(k)

        # print(med)     # 排好序的药物
        path5 = 'data/fangji/'+preName+'/' + preName + '_model_data.csv'
        with  open(path5, 'w') as fw2:
            writer2 = csv.writer(fw2)
            # writer2.writerow(med)
            path6 = 'data/fangji/'+preName+'/' + preName + '_train_data.txt'
            with open(path6, 'r') as fr1:
                reader1 = csv.reader(fr1)

                for i in reader1:
                    # print(i)
                    res = [float(0.0) for i in range(int(dimension))]
                    for j in range(0, len(i), 2):
                        # print(i[j])
                        if i[j] in med:
                            # print(med.index(i[j]))
                            # print(med)
                            # print(i[j])
                            # print(float(i[j+1]))
                            res[med.index(i[j])] = float(i[j+1])

                    writer2.writerow(res)

    print(Tot)
    print(low)
    print(medd)
    print(hig)

def process5_zscore():  # 不需要药典  (z-score)


    preName = ['AS', 'HT', 'HXHY', 'HXZT', 'JP', 'LS', 'MM', 'QRJD', 'TL', 'XZ', 'ZJG', 'ZK', 'ZX', 'ZY']
    all_medical_jiliang = {}
    for na in preName:
        name = 'data/fangji/'+na+'/'+na + '_last_data.csv'
        with open(name,'r') as fr_d:
            reader = csv.reader(fr_d)
            for i in reader:
                # print(i)
                for j in range(0,len(i),2):
                    if i[j].strip()!='':
                        if i[j].strip() not in all_medical_jiliang.keys():
                            all_medical_jiliang.update({i[j].strip():[float(i[j+1])]})
                        else:
                            res = all_medical_jiliang[i[j].strip()]
                            res.append(float(i[j+1]))
                            all_medical_jiliang.update({i[j].strip():res})
    print(all_medical_jiliang)

    all_medical_zscore = {}  # dict(药：[期望，标准差])
    for k,v in all_medical_jiliang.items():
        # print(k,v)

        u = float(sum(v))/len(v)
        biaozhuncha = count_biaozhuncha(v,u)
        all_medical_zscore.update({k:[u,biaozhuncha]})

    print(all_medical_zscore)

    for na in preName:
        print(na)
        path1 = 'data/fangji_zscore/'+na+'/' + na + '_train_data.txt'
        fw = open(path1,'w')
        path2 = 'data/fangji/'+na+'/' + na + '_last_data.csv'
        with open(path2,'r') as fr:
            reader = csv.reader(fr)
            for i in reader:
                medical = []
                for j in range(0,len(i),2):
                    # print(i[j],i[j+1])
                    if i[j].strip() in all_medical_zscore.keys():
                        # print(jiliang[i[j]])
                        val = all_medical_zscore[i[j].strip()]
                        medical.append(i[j].strip())
                        if val[1]!=0:
                            # medical.append( (float(i[j+1])-val[0])/val[1] )
                            medical.append(float(i[j + 1])/ val[1])
                        else:
                            print('hahahhahahhahhahahahhaha')
                            medical.append(0.0)   # 标准差为0 的，直接添加0
                    else:
                        print(i[j])

                # writer.writerow(medical)
                fw.write(','.join([str(k) for k in medical])+'\n')
        fw.close()

        all_medical = set()   # 方剂集下所有的药物
        count_medical = {}

        path3 = 'data/fangji_zscore/'+na+'/' + na + '_train_data.txt'
        with open(path3,'r') as fr1:
            reader1 = csv.reader(fr1)

            nn = 0
            for i in reader1:
                # print(nn+1,i)
                nn +=1
                for j in range(0,len(i),2):
                    all_medical.add(i[j])
                    if i[j].strip() not in count_medical.keys():
                        count_medical.update({i[j].strip():1})
                    else:
                        res = count_medical[i[j].strip()]

                        count_medical.update({i[j].strip():res + 1})

            print('nn:',nn)
        print('方剂集下的所有药物：',len(all_medical))
        print(count_medical)

        med = []
        path4 = 'data/fangji_zscore/'+na+'/' + na + '_count_medical.csv'
        dimension = 0
        with open(path4,'w') as fw1:
            writer1 = csv.writer(fw1)
            for k,v in sorted(count_medical.items(),key=operator.itemgetter(1),reverse=True):
                if float(v)>1:
                    writer1.writerow([k,v])
                    dimension +=1
                    if k not in med:
                        if k.strip()!='':
                            med.append(k)
        print('dimension：',dimension)
        print(med)     # 排好序的药物
        path5 = 'data/fangji_zscore/'+na+'/' + na + '_model_data.csv'
        with  open(path5, 'w') as fw2:
            writer2 = csv.writer(fw2)
            # writer2.writerow(med)
            path6 = 'data/fangji_zscore/'+na+'/' + na + '_train_data.txt'
            with open(path6, 'r') as fr1:
                reader1 = csv.reader(fr1)

                for i in reader1:
                    # print(i)
                    res = [float(0.0) for i in range(int(dimension))]
                    for j in range(0, len(i), 2):
                        # print(i[j])
                        if i[j] in med:
                            # print(med.index(i[j]))
                            # print(med)
                            res[med.index(i[j])] = float(i[j+1])

                    writer2.writerow(res)
def process6(preName):     # 归一化到 0~1


    for na in preName:
        print(na)
        path1 = 'data/fangji/'+na+'/' + na + '_data.csv'
        with open(path1,'w') as fw:
            writer = csv.writer(fw)
            path2 = 'data/fangji/'+na+'/'+ na + '_model_data.csv'
            with open(path2,'r') as  fr:
                reader = csv.reader(fr)
                for i in reader:
                    res = [float(mm) for mm in i]

                    fengmu = float(sum(res))
                    if fengmu==0:
                        # print(1)
                        pass

                    aaa=[]
                    for j in res:
                        if fengmu!=0:
                            # print(float(j/fengmu))
                            # aaa.append(float(j/fengmu))  # 归一化
                            aaa.append(float(j))
                        else:
                            # print(1)
                            aaa.append(0)
                    writer.writerow(aaa)


def process6_zscore():     # 归一化到 0~1

    preName = ['AS', 'HT', 'HXHY', 'HXZT', 'JP', 'LS', 'MM', 'QRJD', 'TL', 'XZ', 'ZJG', 'ZK', 'ZX', 'ZY']

    for na in preName:
        path1 = 'data/fangji_zscore/'+na+'/' + na + '_data.csv'
        with open(path1,'w') as fw:
            writer = csv.writer(fw)
            path2 = 'data/fangji_zscore/'+na+'/'+ na + '_model_data.csv'
            with open(path2,'r') as  fr:
                reader = csv.reader(fr)
                for i in reader:
                    res1 = []
                    res = []
                    for j in i :
                        res1.append(float(j))
                        if float(j)>=0:
                            res.append(float(j))

                        else:
                            # print(float(j))
                            print('ahahah')
                            res.append(-float(j))
                    fengmu = float(sum(res))
                    if fengmu==0:
                        # print(1)
                        print(na)
                        print(i)
                    aaa=[]
                    for j in res1:
                        if fengmu!=0:
                            # print(float(j/fengmu))
                            aaa.append(float(j/fengmu))
                        else:
                            # print(1)
                            aaa.append(0)

                    writer.writerow(aaa)


def process7(preName):
    # preName = ['AS', 'HT', 'HXHY', 'HXZT', 'JP', 'LS', 'MM', 'QRJD', 'TL', 'XZ', 'ZJG', 'ZK', 'ZX', 'ZY']
    for na in preName:
        path1 = 'data/fangji/'+na+'/' + na + '_data2.csv'
        with open(path1, 'w') as fw:
            writer = csv.writer(fw)
            path2 = 'data/fangji/'+na+'/' + na + '_data.csv'
            with open(path2, 'r') as  fr:
                reader = csv.reader(fr)
                data = []
                for i in reader:
                    # print(i)
                    res = []
                    for j in i:

                        if j.strip()=='0.0':
                            res.append('0.0')
                        else:
                            res.append('1.0')
                        res.append(j)  # 第一个为0ne-hot ， 第二个为剂量
                    data.append(res)
                writer.writerows(data)


def process8_onehot(preName):
    # preName = ['AS', 'HT', 'HXHY', 'HXZT', 'JP', 'LS', 'MM', 'QRJD', 'TL', 'XZ', 'ZJG', 'ZK', 'ZX', 'ZY']
    for na in preName:
        print(na)
        path1 = 'data/fangji/' + na + '/' + na + '_onehot_data.csv'
        with open(path1, 'w') as fw:
            writer = csv.writer(fw)
            path2 = 'data/fangji/' + na + '/' + na + '_data.csv'
            with open(path2, 'r') as  fr:
                reader = csv.reader(fr)
                data = []
                for i in reader:
                    # print(i)
                    res = []
                    for j in i:
                        if float(j)>0:

                            res.append(1)
                        else:
                            res.append(0)
                    data.append(res)
                writer.writerows(data)



if __name__=='__main__':

     # process1()    # 整理中药剂量范围
     # process2()  # 整理 中国药典
     dw_conver = {'两': 31.25, '钱': 3.125, '分': 0.3125, '厘': 0.03125, '毫': 0.003125, 'ml': 1, '毫克': 1, '毫升': 1,
                  'mg': 0.001, '钱匕': 2.0,
                  '斤': 500.0, '铢': 0.5789, '克': 1.0, 'g': 1.0, 'kg': 1000.0, '千克': 1000.0, '撮': 10.35, '斗': 2000,
                  '公斤': 1000, '圭': 0.5, '合': 103.5, '斛': 51750, '钧': 6600, '累': 0.057, '升': 500, '石': 103500, '市斤': 500,
                  '市升': 1000, '黍': 0.0083, '桶': 20000, '籥': 10, '龠': 10,
                  '十两': 312.5, '八两': 249.0, '九两': 280.25, '七两': 218.75, '六两': 187.5, '五两': 156.25, '四两': 125.0,
                  '三两': 93.75, '二两': 62.5, '一两': 31.25, '半两': 15.625,
                  '一钱': 3.125, '三钱': 9.375, '一两半': 46.875, '二两半': 78.125,
                  '半钱': 1.5625, '半分': 0.15625, '半斤': 250.0
                  }

     wenjianjia = os.listdir('data/fangji')
     # for preName in wenjianjia:
     #     print(wenjianjia.index(preName))
     #     process3(preName)  #  AS功效下每种药物对应的剂量单位
     #     process4(preName,dw_conver)   # 为每个药物重新标剂量

     # process5(wenjianjia)
     # process5_zscore()

     process6(wenjianjia)
     # process6_zscore()

     # process7(wenjianjia)    # 剂量 + one-hot

     process8_onehot(wenjianjia)



