# coding=utf-8
import re
import data_process
def processAll(allList,preName):
    print('规律总结处理...')
    oneList=[]
    writeList=[]
    for item in allList:
        s=''
        l=[]
        for itemdata in item:
            l.append(itemdata)
            s=s+itemdata+ ','
        print(s[:-1])
        oneList.append(s[:-1])
        writeList.append(l)

    data_process.write_in_csv('../myMedicalModel/modelvsECMSR_Apriori/AprioriResults/ECMA_'+preName+'_Apriori_16.csv', writeList)
    oneSet=list(set(oneList))
    sortList=[]
    for item in oneSet:
        num=oneList.count(item)
        sortList.append([num,item])
    # sortList=sorted(sortList,key=lambda x:x[0],reverse=True)
    #
    # for item in sortList:
    #     print(item)