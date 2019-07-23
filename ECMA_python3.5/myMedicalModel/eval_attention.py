# coding=utf-8
import re
import os
import numpy as np
import tensorflow as tf
import tflearn
from sklearn.utils import shuffle

from myMedicalModel.myModelNolstm_HH import SelfAttentive
from reader import load_csv, VocabDict
import encode_window
import diceEval
import data_process
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from myMedicalModel import processAllList


'''
parse
'''
tf.app.flags.DEFINE_integer('num_epochs',20, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size to train in one step')
tf.app.flags.DEFINE_integer('labels', 2, 'number of label classes')
tf.app.flags.DEFINE_integer('word_pad_length', 20, 'word pad length for training')
tf.app.flags.DEFINE_integer('decay_step', 100, 'decay steps')
tf.app.flags.DEFINE_float('learn_rate', 1e-3, 'learn rate for training optimization')
tf.app.flags.DEFINE_boolean('shuffle', True, 'shuffle data FLAG')
tf.app.flags.DEFINE_boolean('train', True, 'train mode FLAG')
tf.app.flags.DEFINE_boolean('visualize', True, 'visualize FLAG')
tf.app.flags.DEFINE_boolean('penalization', False, 'penalization FLAG')
tf.app.flags.DEFINE_boolean('usePreVector', False, 'preVector FLAG')

FLAGS = tf.app.flags.FLAGS

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
tag_size = FLAGS.labels
word_pad_length = FLAGS.word_pad_length
lr = FLAGS.learn_rate
usePreVector=FLAGS.usePreVector
word_vecs=None

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
def token_parse(iterator):
  for value in iterator:
    return TOKENIZER_RE.findall(value)

tokenizer = tflearn.data_utils.VocabularyProcessor(word_pad_length, tokenizer_fn=lambda tokens: [token_parse(x) for x in tokens])
label_dict = VocabDict()


def plotMetrics(step_losses,epoch_lossacc,preName,preNum):
  pic_path = '../myMedicalModel/forTest/'
  steps = np.arange(0, len(step_losses), 1)

  plot_pic(
    title="Training epoch Loss",
    x_content=steps,
    y_content=step_losses,
    xlabel="Epochs",
    ylabel="Loss ",
    xlim=(0, steps[-1]),
    path=pic_path + "train_loss_"+preName+"_100_0.001_"+preNum+".svg"
  )

  plot_pic(
    title="Training Epoch Acc",
    x_content=steps,
    y_content=epoch_lossacc,
    xlabel="Epochs",
    ylabel="Acc ",
    xlim=(0, steps[-1]),
    path=pic_path + "train_acc_"+preName+"_100_0.001_"+preNum+".svg"
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
    plt.savefig(path, format='svg')
    plt.clf()

def string_parser(arr, fit):
  if fit == False:
    return list(tokenizer.transform(arr))
  else:
    return list(tokenizer.fit_transform(arr))

preNum='1'
preName='QRJD' #69
medical_avg_count=[]
x_label=[]
allList=[]
model = SelfAttentive()
with tf.Session() as sess:
  #load train data
  print('load train data')
  words, tags = load_csv('../data/trainTCM/TCM_train_%s.csv'%preName, target_columns=[0], columns_to_ignore=None,
                         target_dict=label_dict,usePreVector=usePreVector)
  # print('zz',words)
  vocab_list={}
  #  zsy 判断是否适用预训练的词向量 start
  if usePreVector== True:
    input_iter = encode_window.create_document_iter(words)
    vocab = encode_window.encode_dictionary(input_iter)
    vocab_list = vocab.vocabulary_._mapping
    word_vecs = encode_window.load_bin_vec("../medicalVector/model/medicalCorpus_50d.model", vocab_list)
    word_input = encode_window.encode_word(words, vocab,word_pad_length)
  else:
    words = string_parser(words, fit=True)
    if FLAGS.shuffle == True:
      words, tags = shuffle(words, tags)
    word_input = tflearn.data_utils.pad_sequences(words, maxlen=word_pad_length)
  # print('vvv', word_input)
  # zsy 判断是否适用预训练的词向量 end

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
    #clipped_gradients = [tf.clip_by_value(x, -0.5, 0.5) for x in gradients]
    optimizer = tf.train.AdamOptimizer(learn_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    grad_and_vars = tf.gradients(loss, params)
    opt = optimizer.apply_gradients(zip( grad_and_vars, params), global_step=global_step)

  # Start Training
  sess.run(tf.global_variables_initializer())
  total = len(word_input)
  step_print = int((total/batch_size) / 20)
  epoch_losslist=[]
  epoch_lossacc = []

  if FLAGS.train == True:
    print('start training')
    for epoch_num in range(num_epochs):
      epoch_loss = 0
      step_loss = 0
      allnum = 0
      epoch_acc = 0
      for i in range(int(total/batch_size)):
        batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
        train_ops = [opt, loss, learn_rate, global_step,logits]
        if FLAGS.usePreVector== True:
          result = sess.run(train_ops, feed_dict={model.input_pl: batch_input, labels: batch_tags})
        else:
          result = sess.run(train_ops, feed_dict={model.input_pl: batch_input, labels: batch_tags})
        arr = result[4][0].tolist()
        # print(arr.index(max(arr)),batch_tags[0].index(max(batch_tags[0])))
        if arr.index(max(arr))==batch_tags[0].index(max(batch_tags[0])):
          epoch_acc+=1
        allnum+=1
        step_loss += result[1]
        epoch_loss += result[1]
      print('***')
      print('epoch {%s}: (global_step: {%s}), Average Loss: {%s})'%(epoch_num,result[3],(epoch_loss/(total/batch_size))))
      print('***\n')
      epoch_losslist.append(epoch_loss/(total/batch_size))
      epoch_lossacc.append(float(epoch_acc)/allnum)
    saver = tf.train.Saver()
    saver.save(sess, '../myMedicalModel/forTest/0617_model_noLstmHH_r1_%s_epoches%s_num%s.ckpt'%(preName,FLAGS.num_epochs,preNum))
  else:
    saver = tf.train.Saver()
    saver.restore(sess, '../myMedicalModel/forTest/0617_model_noLstmHH_r1_%s_epoches%s_num%s.ckpt'%(preName,FLAGS.num_epochs,preNum))


  # plotMetrics(epoch_losslist,epoch_lossacc,preName,preNum)


  allDice=[]
  lenList=[]
  evalCount = 0

  for a in range(1,51):
      a=float(a)/100
      x_label.append(round(a,2))
      print('start testing')
      words, tags = load_csv('../data/aprioriData/TCM_train_%s.csv' % preName, target_columns=[0], columns_to_ignore=None,
                             target_dict=label_dict)
      words_with_index = string_parser(words, fit=True)
      word_input = tflearn.data_utils.pad_sequences(words_with_index, maxlen=word_pad_length)

      total = len(word_input)
      evalNum = total-1
      rs = 0.
      #load evalData start
      evalData=[]
      evalCav='../data/evalData/%s_evaluate.csv'%preName
      evalList=data_process.read_csv(evalCav)
      for item in evalList:
        evalData.append(item)
      # load evalData end

      if FLAGS.visualize == True:
        f = open('../myMedicalModel/forTest/%s_visualizeTCM_%s_noLSTM_HH_epoches%s_r1_num%s.html'%(preName,preName,FLAGS.num_epochs,preNum), 'w')
        f.write('<html style="margin:0;padding:0;"><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><body style="margin:0;padding:0;">\n')

      for i in range(int(total/batch_size)):
        batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
        result = sess.run([logits, model.B,model.Q], feed_dict={model.input_pl: batch_input, labels: batch_tags})
        #arr保存预测概率
        arr = result[0]
        if not np.argmax(arr[0]):
            preClass=True
        else:
            preClass = False

        for j in range(len(batch_tags)):
          if np.argmax(batch_tags[j])==0:
              if np.argmax(arr[j]) == np.argmax(batch_tags[j]):
                evalCount+=1
          rs+=np.sum(np.argmax(arr[j]) == np.argmax(batch_tags[j]))
        medicalList=[]
        if FLAGS.visualize == True:
          f.write('<div style="margin:15px;">\n')
          #result[1][0]保存的是方剂中每个药物对应的attention因子，具体result[1][0][k][j]取出
          for k in range(len(result[1][0])):
            f.write('\t<p> —— 测试方剂 %s (类标：%s ; 预测类标：%s)：—— </p>\n'%(i, tags[i],preClass))
            f.write('<p style="margin:10px;font-family:SimHei">\n')
            ww = TOKENIZER_RE.findall(words[i*batch_size][0])
            for j in range(word_pad_length):
              if result[1][0][k][j] <a:
                result[1][0][k][j]=0
              alpha = "{:.2f}".format(result[1][0][k][j])
              if len(ww) <= j:
                w = "   "
              else:
                w = ww[j]
                if result[1][0][k][j]>=a:
                    medicalList.append(w)
              f.write('\t<span style="margin-left:3px;background-color:rgba(255,0,0,%s)">%s</span>\n'%(alpha,w))
            f.write('</p>\n')
            if i < evalNum:
              if preClass == True:
                print('配伍评估药组：', medicalList)
                allDice.append(diceEval.evalMedicalDice(medicalList, evalData))
                allList.append(medicalList)
                lenList.append(len(medicalList))
              else:
                allDice.append(0)
            if i < evalNum:
              f.write('\t<b>配伍评估药组： %s ,dice = %s</b>\n' % (','.join(medicalList),allDice[i]))
          f.write('</div>\n')

      if FLAGS.visualize == True:
        f.write('\t<p>Test accuracy: %s</p>\n' % (rs / total))
        f.write('\t<p>该功效下%s个经典方剂(即测试集前%s个方剂) accuracy ：%s</p>\n' % (evalNum,evalNum,evalCount / evalNum))
        f.write('\t<p>该功效下%s个经典方剂 avg-dice : %s</p>\n' % (evalNum,sum(allDice)/evalNum))
        f.write('</body></html>')
        f.close()
      print('Test accuracy(all test data): %s'%(rs/total))
      print('该功效下%s个经典方剂(即测试集前%s个方剂)的accuracy评估：%s' % (evalNum,evalNum,evalCount / evalNum))
      print('allDice,evalCount',allDice,evalCount)
      sumValue=0
      for i in allDice[:evalCount]:
        sumValue+=i
      print('avg-dice:%s'%(sumValue/evalCount))
      print('平均药味数:%s' % (sum(lenList) / evalCount))
      medical_avg_count.append(sum(lenList) / evalCount)
      # print('lenList',lenList)
      # print('medical_avg_count',medical_avg_count)
sess.close()

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 步骤二
plt.figure()
plt.plot(x_label,medical_avg_count)
plt.xlabel("自注意力阈值",fontproperties=font)
plt.ylabel("平均配伍药味数",fontproperties=font)
plt.title("不同阈值下平均药味数统计",fontproperties=font)
plt.savefig("自注意力因子阈值设置0.1_0.5.jpg")


plt.figure()
plt.plot(x_label[:10],medical_avg_count[:10])
plt.xlabel("自注意力阈值",fontproperties=font)
plt.ylabel("平均配伍药味数",fontproperties=font)
plt.title("不同阈值下平均药味数统计",fontproperties=font)
plt.savefig("自注意力因子阈值设置0.01_0.1.jpg")
      # processAllList.processAll(allList,'forTest')
data_process.write_list_in_csv('a.csv',medical_avg_count)
data_process.write_list_in_csv('b.csv',x_label)