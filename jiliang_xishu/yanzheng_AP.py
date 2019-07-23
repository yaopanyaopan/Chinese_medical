# -*-coding:utf-8 -*-
__author__ = '$'
import csv
import math
import os
import apriori
import data_process
import diceEval


wenjianjia = os.listdir('data/fangji')
AP_DICE = []
for preName in wenjianjia:

        # if preName!='补血':
        #     continue

        # print(wenjianjia.index(preName))

        path_ar= 'data/fangji/'+preName+'/'+'Apriori_'+preName+'_data.csv'


        with open(path_ar, 'r') as fr_ar:
                reader_ar = csv.reader(fr_ar)
                data = []
                for i in reader_ar:
                    raw = []
                    for j in i:
                        raw.append(j.strip())
                    data.append(raw)


                print('Apriori_1...')
                apriori_result = apriori.run(preName, data)       # 所有支持度符合的 k频繁项集
                print('Apriori_2...')
                k_apriori_result, sup = data_process.k_top(apriori_result)

                print(preName, k_apriori_result,sup)

                evalueateDataList = []
                evalueatecsv = 'data/fangji/' + preName + '/' + preName + '_evaluate.csv'
                with open(evalueatecsv, 'r') as fr_5:
                    evalueateData = csv.reader(fr_5)
                    for item in evalueateData:
                        item[0] = item[0].replace('﻿', '')
                        evalueateDataList.append(item)

                ap_one_dice = []
                ap_one_dice.append(preName)
                for nn in range(0, 5):
                    if sup[nn] > 0.0:
                        apriori_dice_value = diceEval.evalMedicalDice(k_apriori_result[nn], evalueateDataList)
                    else:
                        apriori_dice_value = '-'
                    ap_one_dice.append(apriori_dice_value)
                    print('top:',nn, apriori_dice_value)

                    AP_DICE.append(ap_one_dice)

R = []
for i in range(1, 6):
    res = 0
    count_1 = 0
    for j in range(0, len(AP_DICE)):
        if AP_DICE[j][i] == '-':
            # print (AP_DICE[j][i])
            pass
        else:
            count_1 += 1
            r = float(AP_DICE[j][i])
            res += r
    if count_1 == 0:
        R.append(0.0)
    else:
        R.append(round(res / count_1, 4))
print('平均值',R)
all_avg = float(sum(R)) / len(R)
print('总体情况',all_avg)
R.insert(0, '平均值')
# writer_ap.writerow(['功效', 'top1', 'top2', 'top3', 'top4', 'top5'])
# writer_ap.writerows(AP_DICE)
# writer_ap.writerow(R)
# writer_ap.writerow(['总体情况', all_avg])
# writer_ap.writerow(ap_QRJD)