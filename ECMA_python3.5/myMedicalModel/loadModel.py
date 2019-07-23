# coding=utf-8
import re
import os
import numpy as np
import tensorflow as tf
import tflearn
from sklearn.utils import shuffle
from tensorflow import reset_default_graph

from myMedicalModel.myModelNolstm_HH import SelfAttentive
from reader import load_csv, VocabDict
import encode_window
import diceEval
import data_process
from matplotlib import pyplot as plt

from myMedicalModel import processAllList

'''
parse
'''
tf.app.flags.DEFINE_integer('num_epochs', 50, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size to train in one step')
tf.app.flags.DEFINE_integer('labels', 2, 'number of label classes')
tf.app.flags.DEFINE_integer('word_pad_length', 50, 'word pad length for training')
tf.app.flags.DEFINE_integer('decay_step', 100, 'decay steps')
tf.app.flags.DEFINE_float('learn_rate', 1e-3, 'learn rate for training optimization')
tf.app.flags.DEFINE_boolean('shuffle', True, 'shuffle data FLAG')
tf.app.flags.DEFINE_boolean('train', True, 'train mode FLAG')
tf.app.flags.DEFINE_boolean('visualize', True, 'visualize FLAG')
tf.app.flags.DEFINE_boolean('penalization', False, 'penalization FLAG')
tf.app.flags.DEFINE_boolean('usePreVector', True, 'preVector FLAG')

FLAGS = tf.app.flags.FLAGS
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
tag_size = FLAGS.labels
word_pad_length = FLAGS.word_pad_length
lr = FLAGS.learn_rate
usePreVector = FLAGS.usePreVector
word_vecs = None

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)

def token_parse(iterator):
    for value in iterator:
        return TOKENIZER_RE.findall(value)

def plotMetrics(step_losses, epoch_lossacc, preName, preNum):
    pic_path = '../myMedicalModel/modelvsECMSR_Apriori/acc_loss_pic/'
    steps = np.arange(0, len(step_losses), 1)
    plot_pic(
        title="Training epoch Loss",
        x_content=steps,
        y_content=step_losses,
        xlabel="Epochs",
        ylabel="Loss ",
        xlim=(0, steps[-1]),
        path=pic_path + "train_loss_" + preName + "_100_0.001_" + str(preNum) + ".png"
    )
    plot_pic(
        title="Training Epoch Acc",
        x_content=steps,
        y_content=epoch_lossacc,
        xlabel="Epochs",
        ylabel="Acc ",
        xlim=(0, steps[-1]),
        path=pic_path + "train_acc_" + preName + "_100_0.001_" + str(preNum) + ".png"
    )

def plot_pic(title, x_content, y_content, xlabel, ylabel, xlim, path):
    print("    - [Info] Plotting metrics into picture " + path)
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = True
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")
    plt.xlim(xlim)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.plot(x_content, y_content)
    plt.xlabel(xlabel, fontsize=13, fontweight='bold')
    plt.ylabel(ylabel, fontsize=13, fontweight='bold')
    plt.savefig(path, format='png')
    plt.clf()

def string_parser(arr, fit):
    if fit == False:
        return list(tokenizer.transform(arr))
    else:
        return list(tokenizer.fit_transform(arr))

#-------------------------------------------------------------------------------------------------
# preNameList = ['QRJD']
preNameList = ['QRJD','HXHY','HT','ZJG','ZX','XZ','LS','MM','JP','ZK','AS','HXZT','ZY','TL']
for preName in preNameList:     # 处理每一个 方剂功效
    attMetricAll = []
    labelList = [[]] * 1000
    allAccList = []
    finalNum = 10
    for preNum in range(0,finalNum+1):     # 重新跑１０次模型　，　取平均值
        tokenizer = tflearn.data_utils.VocabularyProcessor(max_document_length=word_pad_length,
                                                           tokenizer_fn=lambda tokens: [token_parse(x) for x in tokens])
        label_dict = VocabDict()
        reset_default_graph()
        model = SelfAttentive()
        with tf.Session() as sess:
          #load train data
          print('load train data')
          words, tags = load_csv('../data/trainTCM/TCM_train_%s.csv' % preName, target_columns=[0], columns_to_ignore=None,
                                 target_dict=label_dict,usePreVector=usePreVector)
          vocab_list={}
          if FLAGS.shuffle == True:
              words, tags = shuffle(words, tags)
          ws = words
          if usePreVector == True:
              input_iter = encode_window.create_document_iter(words)
              vocab = encode_window.encode_dictionary(input_iter)
              vocab_list = vocab.vocabulary_._mapping
              word_vecs = encode_window.load_bin_vec("../medicalVector/model/medicalCorpus_50d.model", vocab_list)
              # word_vecs = encode_window.load_bin_vec("../medicalVector/model/wv50.model", vocab_list)
              words = encode_window.encode_word(words, vocab, word_pad_length)

              word_input = []
              for hang in range(len(ws)):   # 去掉 pad填补词
                  # print(words[hang])
                  res_words = []
                  for lie in range(len(ws[hang][0].split(' '))):
                      # print(words[hang][lie])
                      if lie < word_pad_length:
                          res_words.append(words[hang][lie])
                  word_input.append(np.array(res_words))
              word_input = list(word_input)

          else:

              words = string_parser(words, fit=True)
              word_input = []
              for hang in range(len(ws)):   # 去掉 pad填补词
                  # print(words[hang])
                  res_words = []
                  for lie in range(len(ws[hang][0].split(' '))):
                      # print(words[hang][lie])
                      if lie < word_pad_length:
                          res_words.append(words[hang][lie])
                  word_input.append(np.array(res_words))
              word_input = list(word_input)
          # print(word_input)
          # build graph
          model.build_graph(n=word_pad_length,usePreVector=usePreVector,vectors=word_vecs)
          # Downstream Application
          with tf.variable_scope('DownstreamApplication'):
              global_step = tf.Variable(0, trainable=False, name='global_step')
              learn_rate = tf.train.exponential_decay(lr, global_step, FLAGS.decay_step, 0.95, staircase=True)
              labels = tf.placeholder('float32', shape=[None, tag_size])
              net = tflearn.fully_connected(model.M, 50, activation='relu')
              logits = tflearn.fully_connected(net, tag_size, activation=None)
              loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1)
              loss = tf.reduce_mean(loss)
              params = tf.trainable_variables()
              optimizer = tf.train.AdamOptimizer(learn_rate)
              grad_and_vars = tf.gradients(loss, params)
              opt = optimizer.apply_gradients(zip( grad_and_vars, params), global_step=global_step)

              attMetric = []
              # Start Training

              sess.run(tf.global_variables_initializer())
              total = len(word_input)
              step_print = int((total / batch_size) / 20)
              epoch_losslist = []
              epoch_lossacc = []

              if FLAGS.train == True:
                print('start training')
                for epoch_num in range(num_epochs):    # 模型迭代周期
                  epoch_loss = 0
                  step_loss = 0
                  allnum = 0
                  epoch_acc = 0
                  for i in range(int(total/batch_size)):     # 遍历训练集
                    batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
                    train_ops = [opt, loss, learn_rate, global_step,logits]
                    if FLAGS.usePreVector== True:
                      result = sess.run(train_ops, feed_dict={model.input_pl: batch_input, labels: batch_tags})
                    else:
                      result = sess.run(train_ops, feed_dict={model.input_pl: batch_input, labels: batch_tags})
                    arr = result[4][0].tolist()        # 矩阵　＝》　列表
                    if arr.index(max(arr))==batch_tags[0].index(max(batch_tags[0])):
                      epoch_acc +=1
                    allnum+=1
                    step_loss += result[1]
                    epoch_loss += result[1]

                  print('epoch {%s}: (global_step: {%s}), Average Loss: {%s})' % (epoch_num , result[3] , (epoch_loss/(total/batch_size))))
                  # print('***')
                  epoch_losslist.append(epoch_loss/(total/batch_size))
                  epoch_lossacc.append(float(epoch_acc)/allnum)
                saver = tf.train.Saver()               # 1 轮迭代５０次后保存模型
                saver.save(sess, '../myMedicalModel/modelvsECMSR_Apriori/model/model_noLstmHH_r1_%s_epoches%s_num%s.ckpt'%(preName,FLAGS.num_epochs,preNum))
              else:
                saver = tf.train.Saver()
                saver.restore(sess, '../myMedicalModel/modelvsECMSR_Apriori/model/model_noLstmHH_r1_%s_epoches%s_num%s.ckpt'%(preName,FLAGS.num_epochs,preNum))

              plotMetrics(epoch_losslist , epoch_lossacc , preName , preNum)     # 画出　损失在周期中的变化

              allDice=[]
              lenList=[]
              evalCount = 0
              pos10Count=0
              a = 0.1                     # 自注意力阀值  zsy

              print('start testing')
              print('功效：',preName,'次数：',preNum)
              allList = []
              words, tags = load_csv('../data/aprioriData/TCM_train_%s.csv'%preName, target_columns=[0], columns_to_ignore=None, target_dict=label_dict)
              length = len(words)
              words = words[:length-1]
              tags = tags[:length-1]
              ws = words

              if usePreVector==False:
                  words_with_index = string_parser(words , fit=True)
                  # print(words_with_index)
              else:
                  # word_vecs = encode_window.load_bin_vec("../medicalVector/model/medicalCorpus_50d.model", vocab_list)
                  words_with_index = encode_window.encode_word(words, vocab, word_pad_length)
                  # print(words_with_index)
              word_input = []
              for hang in range(len(ws)):  # 去掉 pad填补词
                  res_words = []
                  for lie in range(len(ws[hang][0].split(' '))):
                      if lie < word_pad_length:
                          res_words.append(words_with_index[hang][lie])
                  # print(res_words)
                  word_input.append(np.array(res_words))
              word_input = list(word_input)
              # print(word_input)

              total = len(word_input)           # 测试数据量
              evalNum = total-1
              rs = 0.
              #-----------------------load evalData start----------------
              evalData=[]
              evalCav = '../data/evalData/%s_evaluate.csv' % preName
              evalList = data_process.read_csv(evalCav)
              for item in evalList:
                evalData.append(item)
              # load evalData end
              k_count = 0
              if FLAGS.visualize == True and preNum < finalNum:
                  f = open('../myMedicalModel/modelvsECMSR_Apriori/html/%s_visualizeTCM_%s_noLSTM_HWH_epoches%s_r1_num%s.html' % (
                  preName, preName, FLAGS.num_epochs, preNum), 'w')
                  f.write(
                      '<html style="margin:0;padding:0;"><meta http-equiv="Content-Type" content="text/html; charset=GBK"><body style="margin:0;padding:0;">\n')
                  for i in range(int(total / batch_size)):   # 遍历每一条数据
                      batch_input, batch_tags = (
                      word_input[i * batch_size:(i + 1) * batch_size], tags[i * batch_size:(i + 1) * batch_size])
                      result = sess.run([logits, model.B, model.Q], feed_dict={model.input_pl: batch_input, labels: batch_tags})
                      # arr保存预测概率
                      arr = result[0].tolist()
                      thisLabel = sess.run(tf.nn.softmax(arr[0]))
                      if labelList[k_count]==[]:
                          labelList[k_count] = thisLabel
                      else:
                          labelList[k_count][0] += thisLabel[0]
                          labelList[k_count][1] += thisLabel[1]
                      k_count +=1
                      if not np.argmax(arr[0]):
                          preClass = True
                      else:
                          preClass = False
                      for j in range(len(batch_tags)):
                          # 评估10个经典正例的accuracy——>evalCount/10
                          if pos10Count < evalNum:
                              if np.argmax(batch_tags[j]) == 0:
                                  if np.argmax(arr[j]) == np.argmax(batch_tags[j]):
                                      evalCount += 1          #  预测是否有该功效 正确
                              pos10Count += 1
                          # end
                          rs += np.sum(np.argmax(arr[j]) == np.argmax(batch_tags[j]))
                      medicalList = []
                      if FLAGS.visualize == True and preNum <finalNum:  # 画 html
                          f.write('<div style="margin:15px;">\n')
                          # result[1][0]保存的是方剂中每个药物对应的attention因子，具体result[1][0][k][j]取出
                          for k in range(len(result[1][0])):
                              f.write('\t<p> —— 测试方剂 %s (类标：%s ; 预测类标：%s)：—— </p>\n' % (i, tags[i], preClass))
                              f.write('<p style="margin:10px;font-family:SimHei">\n')

                              ww = words[i*batch_size][0].split(' ')
                              # print(ww)
                              thisMetic = []

                              for j in range(len(batch_input[0])):
                                  thisMetic.append(round(result[1][0][k][j], 3))
                                  # print(result[1][0][k][j])
                                  if result[1][0][k][j] < a:
                                      result[1][0][k][j] = 0
                                  alpha = "{:.2f}".format(result[1][0][k][j])
                                  if len(ww) <= j:
                                      w = "   "
                                  else:
                                      w = ww[j]
                                      if result[1][0][k][j] >= a:
                                          medicalList.append(w)
                                  f.write(
                                      '\t<span style="margin-left:3px;background-color:rgba(255,0,0,%s)">%s</span>\n' % (alpha, w))
                              f.write('</p>\n')
                              if i < evalNum:
                                  if preClass == True:       # 对于模型预测正确的做 dice评估
                                      # print('配伍评估药组：', medicalList)
                                      allDice.append(diceEval.evalMedicalDice(medicalList, evalData))
                                      allList.append(medicalList)
                                      lenList.append(len(medicalList))
                                  else:
                                      allDice.append(0)
                              if i < evalNum:
                                  f.write('\t<b>配伍评估药组： %s ,dice = %s</b>\n' % (','.join(medicalList), allDice[i]))
                              attMetric.append(thisMetic)    # 自注意力矩阵(10行相拼)
                          f.write('</div>\n')

                  if FLAGS.visualize == True and preNum < finalNum:
                      f.write('\t<p>Test accuracy: %s</p>\n' % (rs / total))
                      f.write('\t<p>该功效下%s个经典方剂(即测试集前%s个方剂) accuracy ：%s</p>\n' % (evalNum, evalNum, evalCount / evalNum))
                      f.write('\t<p>该功效下%s个经典方剂 avg-dice : %s</p>\n' % (evalNum, sum(allDice) / evalNum))
                      f.write('</body></html>')
                      f.close()
                      data_process.write_in_csv('../myMedicalModel/modelvsECMSR_Apriori/metircsResults/attMetric'+preName+str(preNum)+".csv",attMetric)
                      if attMetricAll==[]:
                        attMetricAll = attMetric
                      else:
                          for x,itemx in enumerate(attMetricAll):
                              for y,itemy in enumerate(itemx):
                                  attMetricAll[x][y] = attMetricAll[x][y] + attMetric[x][y]
              ##########################################################最终一次结果显示
              if FLAGS.visualize == True and preNum==finalNum:     #　第 11次结果(我们想要的)
                  f = open('../myMedicalModel/modelvsECMSR_Apriori/html/final_%s_visualizeTCM_%s_noLSTM_HWH_epoches%s_r1_num%s.html'%(preName,preName,FLAGS.num_epochs,preNum), 'w')
                  f.write('<html style="margin:0;padding:0;"><meta http-equiv="Content-Type" content="text/html; charset=GBK"><body style="margin:0;padding:0;">\n')
                  k_count = 0
                  for i in range(int(total/batch_size)):
                    batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
                    result = sess.run([logits, model.B,model.Q], feed_dict={model.input_pl: batch_input, labels: batch_tags})
                    for j in range(len(batch_tags)):
                        if np.argmax(batch_tags[j]) == 0:
                            if np.argmax(labelList[k_count]) == np.argmax(batch_tags[j]):
                                evalCount += 1
                    medicalList = []
                    if FLAGS.visualize == True and preNum==finalNum:
                      f.write('<div style="margin:15px;">\n')
                      #result[1][0]保存的是方剂中每个药物对应的attention因子，具体result[1][0][k][j]取出
                      for k in range(len(result[1][0])):
                        if not np.argmax(labelList[k_count]):
                            preClass = True
                        else:
                            preClass = False
                        f.write('\t<p> —— 测试方剂 %s (类标：%s ; 预测类标：%s)：—— </p>\n'%(i, tags[i],preClass))
                        f.write('<p style="margin:10px;font-family:SimHei">\n')
                        ww = words[i*batch_size][0].split(' ')

                        for j in range(len(batch_input[0])):
                          if (attMetricAll[k_count][j]/finalNum) < a:    #　１０个自注意力取平均值
                              color = 0
                          else:
                              color = attMetricAll[k_count][j]/finalNum
                          alpha = "{:.2f}".format(color)
                          if len(ww) <= j:
                            w = "   "
                          else:
                            w = ww[j]
                            if color >= a:
                                medicalList.append(w)
                          f.write('\t<span style="margin-left:3px;background-color:rgba(255,0,0,%s)">%s</span>\n'%(alpha,w))
                        f.write('</p>\n')
                        if i < evalNum:
                          if preClass == True:
                            # print('配伍评估药组：', medicalList)
                            allDice.append(diceEval.evalMedicalDice(medicalList, evalData))
                            allList.append(medicalList)
                            lenList.append(len(medicalList))
                          else:
                            allDice.append(0)
                        if i < evalNum:
                          f.write('\t<b>配伍评估药组： %s ,dice = %s</b>\n' % (','.join(medicalList) , allDice[i]))
                      k_count += 1
                      f.write('</div>\n')


                  if FLAGS.visualize == True and preNum==finalNum:
                    f.write('\t<p>Test accuracy: %s</p>\n' % (rs / total))
                    f.write('\t<p>该功效下%s个经典方剂(即测试集前%s个方剂) accuracy ：%s</p>\n' % (evalNum ,evalNum ,evalCount / evalNum))
                    f.write('\t<p>该功效下%s个经典方剂 avg-dice : %s</p>\n' % (evalNum ,sum(allDice) / evalNum))
                    f.write('</body></html>')
                    f.close()
                    processAllList.processAll(allList,preName)


              print('Test accuracy(all test data): %s'%(rs/total))
              print('该功效下%s个经典方剂(即测试集前%s个方剂)的accuracy评估：%s' % (evalNum ,evalNum ,evalCount / evalNum))
              print('allDice,evalCount',allDice , evalCount)
              sumValue=0
              for i in allDice[:evalCount]:
                sumValue += i
              print('avg-dice:%s'%(sumValue / evalCount))
              print('平均药味数:%s' % (sum(lenList) / evalCount))
              print('accList:',epoch_lossacc)
              allAccList.append(epoch_lossacc)
        sess.close()
        del sess
    data_process.write_list_in_csv('../myMedicalModel/modelvsECMSR_Apriori/metircsResults/acc_'+preName+".csv",allAccList)