# Self-Attentive-Tensorflow



 noLstm_WH_model  (此模块为ECMA-Attention1(药物之间没有做线性变换)相关代码，用于对比)
 myMedicalModel  (主要代码实现)
 data  （实验数据）
 medicalVector  （预训练好的药向量，但是这个药向量质量不好）
```
数据部分说明：
/data

   aprioriData  (此模块为ECMA-Attention1相关代码，用于模型对比实验)
   trainTCM  (各功效训练数据)
   evalData （各功效标准配伍集，用于dice评估）
   profession  （用于人工对比实验）
   testData（用于模型对比实验数据）

主要功能代码说明（可以直接运行）：
./myMedicalModel

     loadModel.py  (每次跑10次求对应注意力均值得到药组进行dice评估，在myMedicalModel—>AprioriResults下查看得到的药组项集，然后转用Apriori算法得到一组配伍。)
     profession_eval.py 
     eval_attention.py  （设置自注意力阈值,绘图代码）
     























