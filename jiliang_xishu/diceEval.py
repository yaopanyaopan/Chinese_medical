#--*--coding=utf-8 --*--
def evalMedicalDice(medicalList,evalData):
    dice=0
    # print(medicalList)
    # print(evalData)
    for i,item in enumerate(evalData):
        calnum=0

        for itemdata in item:
            if i==0:
                itemdata=itemdata.replace('﻿', '')
                # print(itemdata)
            for medical in medicalList:
                if itemdata.find(medical)>-1:
                    calnum+=1
        value=2*calnum/(len(item)+len(medicalList))
        if value>dice:
            dice=value
        if dice==1:
            break
    return round(dice,4)







