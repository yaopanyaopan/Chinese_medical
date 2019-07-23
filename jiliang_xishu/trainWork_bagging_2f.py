# coding=utf-8
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
from sklearn import linear_model
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *
import random
import csv
import diceEval
import apriori
import data_process

def loadData(preName):
   #处理好的数据集 已经正负例平衡
    train_x = []
    train_y = []
    sumZeroList=[]

    presCsvname = 'data/fangji/'+preName+'/'+preName+'_data.csv'  # 只有药物做了归一化
    funcCsvname = 'data/fangji/'+preName+'/'+'FuncFeature_'+preName+'.csv'
    # data = data_process.read_csv( presCsvname )
    fr = open(presCsvname,'r')
    data = csv.reader(fr)

    fr1 = open(funcCsvname,'r')
    labeldata = csv.reader(fr1)    # 0/1标签，判断是否含当前方剂功效

    check = 0
    for i in data:
        if check==0:
            i[0] = i[0].replace('﻿', '')
        i = [float(item) for item in i]    # 转换输入为数字
        train_x.append(i)
        check += 1

    check = 0
    for j in labeldata:
        if check==0:
            j[0] = j[0].replace('﻿', '')
        # print(j[0])
        train_y.append(float(j[0]))      # 转换标签为数字
        check += 1

    # 随机打乱数据集
    index = [i for i in range(len(train_x))]
    random.shuffle(index)    # 随机打乱索引 index
    num = 0
    my_x = [1] * len(train_x)
    my_y = [1] * len(train_y)
    for item in index :           # 遍历所有打乱索引， 构造训练集
        my_x[num] = train_x[item]
        my_y[num] = train_y[item]
        num +=1

    return mat(my_x), my_y

def loadData1(preName):
   #处理好的数据集 已经正负例平衡
    train_x = []
    train_y = []
    sumZeroList=[]

    presCsvname = 'data/fangji/'+preName+'/'+preName+'_onehot_data.csv'  # 只有药物做了归一化
    funcCsvname = 'data/fangji/'+preName+'/'+'FuncFeature_'+preName+'.csv'
    # data = data_process.read_csv( presCsvname )
    fr = open(presCsvname,'r')
    data = csv.reader(fr)

    fr1 = open(funcCsvname,'r')
    labeldata = csv.reader(fr1)    # 0/1标签，判断是否含当前方剂功效

    check = 0
    for i in data:
        if check==0:
            i[0] = i[0].replace('﻿', '')
        i = [float(item) for item in i]    # 转换输入为数字
        train_x.append(i)
        check += 1

    check = 0
    for j in labeldata:
        if check==0:
            j[0] = j[0].replace('﻿', '')
        # print(j[0])
        train_y.append(float(j[0]))      # 转换标签为数字
        check += 1

    # 随机打乱数据集
    index = [i for i in range(len(train_x))]
    random.shuffle(index)    # 随机打乱索引 index
    num = 0
    my_x = [1] * len(train_x)
    my_y = [1] * len(train_y)
    for item in index :           # 遍历所有打乱索引， 构造训练集
        my_x[num] = train_x[item]
        my_y[num] = train_y[item]
        num +=1

    return mat(my_x), my_y

if __name__ == '__main__':
    ## step 1: load data
    print ("step 1: load data...")
    # 功效【手动修改】 除了这两个，还要改数据库名称！！！！start

    # ----------------------------------------
    top_medical = {}

    AP_DICE = []
    DICE = []
    for tt in range(14):
        ## step 1: load data
        print("step 1: load data...")
        # 功效【手动修改】 除了这两个，还要改数据库名称！！！！start

        # ----------------------------------------
        if tt == 0:
            preName = 'QRJD'
            dimension = '582'
            function = '清热解毒'  # 中文

        # 活血化瘀
        elif tt == 1:
            preName = 'HXHY'
            dimension = '355'
            function = '活血化瘀'  # 中文
        # 化痰
        elif tt == 2:
            preName = 'HT'
            dimension = '687'
            function = '化痰'  # 中文
        # 壮筋骨
        elif tt == 3:
            preName = 'ZJG'
            dimension = '347'
            function = '壮筋骨'  # 中文
        # 止血
        elif tt == 4:
            preName = 'ZX'
            dimension = '359'
            function = '止血'  # 中文
        # 消肿
        elif tt == 5:
            preName = 'XZ'
            dimension = '549'
            function = '消肿'  # 中文
        # 利湿
        elif tt == 6:
            preName = 'LS'
            dimension = '375'
            function = '利湿'  # 中文
        # 明目
        elif tt == 7:
            preName = 'MM'
            dimension = '413'
            function = '明目'  # 中文
        # 健脾
        elif tt == 8:
            preName = 'JP'
            dimension = '596'
            function = '健脾'  # 中文
        # 止咳
        elif tt == 9:
            preName = 'ZK'
            dimension = '322'
            function = '止咳'  # 中文
        # 安神
        elif tt == 10:
            preName = 'AS'
            dimension = '396'
            function = '安神'  # 中文
        # 活血止痛
        elif tt == 11:
            preName = 'HXZT'
            dimension = '209'
            function = '活血止痛'  # 中文
        # 止痒
        elif tt == 12:
            preName = 'ZY'
            dimension = '314'
            function = '止痒'  # 中文
        # 通络
        else:
            preName = 'TL'
            dimension = '327'
            function = '通络'  # 中文
    # ----------------------------------------


        # 功效【手动修改】 end
        print ('获取数据')
        data_x, data_y = loadData(preName)
        X_train = data_x
        y = data_y

        data_x1, data_y1 = loadData1(preName)
        X_train1 = data_x1
        y1 = data_y1

        numSamples, numFeatures = shape(X_train)
        print ('样本数量：', numSamples)
        print ('每个实例有多少维特征：', numFeatures)
        num = 1

    #----------------开始训练------------------------
        for i in range(4,5):
            maxiter = i * 50
            print ('迭代次数设置：', maxiter)
            fw = open('logistic_L1_liblinear_dosageMedical.csv', 'w')
            writer = csv.writer(fw)

            for k in range(1,2):
                for j in range(1, 2):
                    print ("step 2: training...")
                    c0 = 0.98
                    c1 = 0.25
                    # c = float(1) / j
                    # if k == 1:
                    #     c = c * 10
                    # if k == 2:
                    #     c = c * 100
                    # if k == 3:
                    #     c = c * 1000
                    # if k == 4:
                    #     c = c * 10000
                    lamda = round((float(1) / c0),5)      # 正则化因子
                    print ('正则化因子lamda设置：', lamda)
                # 使用　坐标轴下降法(可同时　优化Ｌ１，Ｌ２正则化)
                    logistic_lasso = linear_model.LogisticRegression(C=c0 , penalty='l1',fit_intercept =False,max_iter=maxiter,solver='liblinear',tol=1e-20).fit(X_train,y)
                    logistic_lasso1 = linear_model.LogisticRegression(C=c1, penalty='l1', fit_intercept=False,
                                                                     max_iter=maxiter, solver='liblinear',
                                                                     tol=1e-20).fit(X_train1, y1)

                    # 此处 alpha 为通常值 #fit 把数据套进模型里跑

                    loss = logistic_lasso

                    coef = pd.Series( logistic_lasso.coef_.ravel() )  # 模型对应的参数权重
                    coef1 = pd.Series(logistic_lasso1.coef_.ravel())  # 模型对应的参数权重

                    n_iter = logistic_lasso.n_iter_[0]       # 模型实际　迭代次数
                    # .coef_ 可以返回经过学习后的所有 feature 的参数。
                    count_zero = 0
                    for i in coef:
                        if abs(i)==0:
                            count_zero += 1
                    count_nonzero = 2*int(dimension) - count_zero   # Ｌ１正则化后，权重非０个数
                    print ('由LASSO约束后变为0的参数个数以及实际迭代次数：', count_zero , n_iter)
                    print ("step 3: testing...")
                    accuracy = logistic_lasso.score(X_train , y)   # 返回　平均准确率
                    print ('The classify accuracy is: %.3f%%' % (accuracy * 100))
                    ## step 4: show and write the result
                    print ("step 4: write the result in csv files...")
                    writecsvname = '../%sresults/l1_weight_dosageAll_%s_' % (preName, preName)
                    writecsvname = writecsvname + str(maxiter) +'_'+ str(lamda) + '.csv'
                    # data_process.write_list_in_csv(writecsvname, coef)
                    print ("step 5: find medical group and write the resultsdata...")
                    weightdata0 = coef    # 对应参数
                    weightdata1 = coef1  # 对应参数
                    csvname = 'data/fangji/'+preName+'/'+preName+'_count_medical.csv'
                    fr3 = open(csvname,'r')
                    medicaldata = csv.reader(fr3)
                    medicallist = []
                    importantMedical = []
                    weightlist = []
                    weightlist0 = []
                    weightlist1 = []
                    for item in medicaldata:
                        # print item[0]
                        medicallist.append(item[0].replace('﻿', ''))


                    num1 = 0
                    # print '再次确认一下参数个数：',len(weightdata)

                    # for item in weightdata:
                    #         print(item)
                    #         zz = []
                    #         zz.append(num1)
                    #         zz.append(item)
                    #         weightlist.append(zz)
                    #         num1 += 1
                    for item in range(0,len(weightdata0)):
                            # print(weightdata[item],weightdata[item])
                            zz = []
                            zz.append(num1)
                            zz.append(float(weightdata0[item]))
                            weightlist0.append(zz)
                            num1 += 1
                    num1 = 0
                    for item in range(0,len(weightdata1)):
                            # print(weightdata[item],weightdata[item])
                            zz = []
                            zz.append(num1)
                            zz.append(float(weightdata1[item]))
                            weightlist1.append(zz)
                            num1 += 1
                    # 合并两个模型的权重
                    num1 = 0
                    for hang in range(len(weightdata0)):
                            zz = []
                            zz.append(num1)
                            zz.append(float(weightlist0[hang][1]) + float(weightlist1[hang][1]))
                            weightlist.append(zz)
                            num1 += 1

                    weightlist = sorted(weightlist, key=lambda x: x[1], reverse=True)   # 排序模型参数对应的药物

                    forEvaData = []
                    text = ""
                    res_medical = []
                    for i in range(0, 20):  # 权重最高的前２０个药物

                        if len(res_medical) < 5 and weightlist[i][1] > 0.0:
                            res_medical.append(medicallist[weightlist[i][0]])
                        zz = []
                        zz.append(medicallist[weightlist[i][0]])
                        zz.append(weightlist[i][1])

                        forEvaData.append( medicallist[weightlist[i][0]] )
                        text = text + medicallist[weightlist[i][0]] + ',' + str(weightlist[i][1]) + ';'

                        importantMedical.append(zz)
                    top_medical.update({preName: res_medical})
                    print ('当功效为‘%s’时，占主导作用的药物组合是：\n' % function)
                    count = 0
                    for item in importantMedical:
                        print ('药物%d:' % (count + 1), item[0], item[1])
                        count += 1

                    print ('step 6: medical evaluating....')
                    # evalueatecsv='../formulaData_1/HXHY_evaluate.csv'
                    evalueatecsv = 'data/fangji/'+preName+'/'+preName+'_evaluate.csv'

                    fr4 = open(evalueatecsv,'r')
                    evalueateData = csv.reader(fr4)

                    evalueateDataList = []
                    for item in evalueateData:
                        item[0] = item[0].replace('﻿', '')
                        for i in range(len(item)):
                            # item[i] = item[i].decode('utf8')
                            pass
                        evalueateDataList.append(item)

                    checklist = [0] * 10

                    # 依次从1到n计算每个标准评估配伍的 MAX（p_avg）
                    # 取标准评估集里的一个配伍集进行评估
                    standardNum = 1
                    bestmarktext = ""
                    bestmark = 0
                    for itemSet in evalueateDataList:
                        si = 0
                        piSum = 0
                        evaluateNum = len(itemSet)
                        # print '第 %d 个评估的标准配伍（itemSet):'%standardNum,itemSet,evaluateNum
                        # 控制权重Top N 的取值【1,20】
                        for n in range(0, 10):
                            # print '设置权重取值top n:',n+1
                            # (n,n+1)这样设置是为了 过滤已经判别过的药物
                            for i in range(n, n + 1):

                                for item in itemSet:     # 每次要与标准药物一一匹配
                                    # print forEvaData[n],item,forEvaData[i].find(item)
                                    if forEvaData[n].find(item) > -1:      # 如果模型中某一药物在评估集中
                                        si += 1
                                        piSum = piSum + float(si) / (n + 1)   # 查准率
                                        # print 'zzzz',pavg
                            pavg = round(float( piSum / evaluateNum),3)   # 分母是一个标准评估组里的药物集
                            bestmark = 0
                            if pavg > checklist[n]:
                                bestmark = standardNum
                                bestmarktext = bestmarktext + str(standardNum) + ',' + str(n + 1) + ';'
                                checklist[n] = pavg
                                # print 'check',checklist,bestmark,n
                        standardNum += 1
                    # checklist存放随着top n的取值变化的最高平均查准率，bestmark存放评估效果最好的那个标准配伍集
                    print ('check finally：')
                    print ('随着top n(1 to 10)的取值变化的最高平均查准率:', checklist)
                    print ('评估效果最好的标准配伍集序号:' , bestmark)
                    print ('本次实验top 5的性能评估值为：' , max(checklist[:4]))
                    print ('本次实验top 10的性能评估值为：' , max(checklist[:9]))
                    print ('\n')

                    print('Dice评价')
                    one_dice = []
                    one_dice.append(preName)
                    for mm in range(1,6):
                        # if weightlist[mm - 1][1] > 0.0:
                        dice_value = diceEval.evalMedicalDice(forEvaData[:mm], evalueateDataList)

                        one_dice.append(dice_value)
                        print('top:', mm, dice_value)
                    if preName == 'QRJD':
                        QRJD = one_dice
                    else:
                        DICE.append(one_dice)

                        #---------------保存模型实验数据-------------------------
                    a = [function, "ave=60", dimension, count_nonzero, n_iter, lamda, accuracy * 100, max(checklist[:4]),
                         max(checklist[:9]), text,
                         checklist[0], checklist[1], checklist[2], checklist[3], checklist[4], checklist[5], checklist[6],
                         checklist[7], checklist[8], checklist[9]]

                    writer.writerow(a)

                    # # -----绘图！！！选头尾各10条，.sort_values() 可以将某一列的值进行排序。-----
                    # coef_pic = pd.Series(logistic_lasso.coef_.ravel(),index= medicallist)
                    # # print coef_pic.index,coef_pic.values
                    # nameList=[]
                    # valueList=[]
                    # kk=0
                    # for i in coef_pic.values:
                    #     if abs(i)==0:
                    #         valueList.append(round(i,3))
                    #         nameList.append(coef_pic.index[kk])
                    #         # print i,coef_pic.index[kk]
                    #     kk+=1
                    # for item in nameList:
                    #     coef_pic.pop(item)
                    # # 画所有参数
                    # matplotlib.rcParams['axes.unicode_minus'] = False
                    # matplotlib.rcParams['figure.figsize'] = (12.0, 8.0)
                    # coef_pic.plot(kind = "bar", width = 0.8)
                    # plt.title("Herb Weights in the Sparse Model")
                    # plt.xlabel('Herb')
                    # plt.ylabel('Weights')
                    # plt.show()


                    num += 1
                fw.close()
    # 记录 dice结果
    R = []
    for i in range(1, 6):
        res = 0

        for j in range(0, len(DICE)):

                r = float(DICE[j][i])
                res += r
        R.append(round(res / len(DICE), 4))

    print(QRJD)
    print('平均值', R)
    all_avg = float(sum(R)) / len(R)
    print('总体情况', all_avg)
    R.insert(0, '平均值')

    with open('dice_result.csv', 'w') as fw:
        writer_r = csv.writer(fw)
        writer_r.writerow(['功效', 'top1', 'top2', 'top3', 'top4', 'top5'])
        writer_r.writerows(DICE)
        writer_r.writerow(R)
        writer_r.writerow(['总体情况', all_avg])
        writer_r.writerow(QRJD)

    print(top_medical)
    # ------- 关联规则-------

    with open('apriori_dice_result.csv', 'w') as fw_ar:
        writer_ap = csv.writer(fw_ar)
        for k, v in top_medical.items():

            path_ar = 'data/fangji/' + k + '/' + 'Apriori_' + k + '_data.csv'
            with open(path_ar, 'r') as fr_ar:
                reader_ar = csv.reader(fr_ar)
                data = []
                for i in reader_ar:
                    raw = []

                    for j in i:
                        if j.strip() in v:
                            raw.append(j.strip())
                    data.append(raw)
                # print(data)
                apriori_result = apriori.run(k, data)  # 所有支持度符合的 k频繁项集
                k_apriori_result, sup = data_process.k_top(apriori_result)

                # print(k, k_apriori_result,sup)

                evalueateDataList = []
                evalueatecsv = 'data/fangji/' + k + '/' + k + '_evaluate.csv'
                with open(evalueatecsv, 'r') as fr_5:
                    evalueateData = csv.reader(fr_5)
                    for item in evalueateData:
                        item[0] = item[0].replace('﻿', '')
                        for i in range(len(item)):
                            # item[i] = item[i].decode('utf8')
                            pass
                        evalueateDataList.append(item)

                ap_one_dice = []
                ap_one_dice.append(k)
                for nn in range(0, 5):
                    if sup[nn] > 0.0:
                        apriori_dice_value = diceEval.evalMedicalDice(k_apriori_result[nn], evalueateDataList)
                    else:
                        apriori_dice_value = '-'
                    ap_one_dice.append(apriori_dice_value)
                    # print('top:',nn, apriori_dice_value)
                if k == 'QRJD':
                    ap_QRJD = ap_one_dice
                else:
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
            R.append(round(res / count_1, 4))
        # print(ap_QRJD)
        # print('关联规则：平均值', R)
        all_avg = float(sum(R)) / len(R)
        # print('关联规则：总体情况', all_avg)
        R.insert(0, '平均值')
        writer_ap.writerow(['功效', 'top1', 'top2', 'top3', 'top4', 'top5'])
        writer_ap.writerows(AP_DICE)
        writer_ap.writerow(R)
        writer_ap.writerow(['总体情况', all_avg])
        writer_ap.writerow(ap_QRJD)







