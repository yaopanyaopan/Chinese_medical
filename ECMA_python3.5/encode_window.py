#--*--coding=utf-8 --*--
import numpy as np
import pickle
import gensim, os
import tensorflow as tf
from gensim.models import Word2Vec

def create_document_iter(tokens):
    for doc in tokens:
        # print(doc[0].strip())
        # raw_doc = ""
        # for word in doc:
        #     raw_doc += " " + word
        yield doc[0].strip()

def encode_dictionary(input_iter, min_frequence=0, max_document_length=20000):
    # print('input_iter',input_iter)
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length,
        min_frequence)
    vocab_processor.fit(input_iter)
    return vocab_processor

def encode_word(tokens,vocab_processor,word_pad_length=20):
    words =[]
    unk = vocab_processor.vocabulary_._mapping["<UNK>"]
    j = 0
    for doc in tokens:
        doc=doc[0].strip()
        doc = doc.split(" ")
        sub_words = []
        for tok in np.arange(len(doc)):
            sub_words.append(vocab_processor.vocabulary_._mapping.get(doc[tok], unk))
        if len(sub_words) < word_pad_length:
            while len(sub_words) < word_pad_length:
                sub_words.append(0)
        if len(sub_words) > word_pad_length:
            sub_words = sub_words[:word_pad_length]
        words.append(sub_words)
        j += 1


    return words

def load_bin_vec(fname, vocab):
    """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
    word_vecs = np.zeros((len(vocab), 50))
    count = 0
    mis_count = 0
    vocab_bin = Word2Vec.load(fname).wv
    for word in vocab:
        if  word in vocab_bin:
            count += 1
            word_vecs[vocab[word]]=(vocab_bin[word])
        else:
            # print('not found',word)
            mis_count += 1
            word_vecs[vocab[word]] = (np.random.uniform(-0.25, 0.25, 50))
    print("found %d" %count)
    print("not found %d" %mis_count)
    return word_vecs

def add_unknown_words(vocab):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    word_vecs = np.zeros((len(vocab), 50))
    for word in vocab:
            word_vecs[vocab[word]] = (np.random.uniform(-0.25, 0.25, 50))
    return word_vecs

if __name__ == "__main__":
    # 通络
    preName = 'TL'
    dimension = '327'
    function = '通络'  # 中文

    tokens = pickle.load(open("../wordEmbeddingData/bin/TCM_em_%s.bin" % preName, "rb"))
    input_iter = create_document_iter(tokens)
    vocab = encode_dictionary(input_iter)
    vocab_list = vocab.vocabulary_._mapping

    word_vecs = load_bin_vec("./model/medicalCorpus_50d.model", vocab_list)
    pickle.dump(word_vecs, open("../wordEmbeddingData/vector/vector_cbow_%s.bin"%preName, "wb"))

    del word_vecs

    words = encode_word(tokens, vocab)
    pickle.dump(words, open("../wordEmbeddingData/wordIndex/wordindex_cbow_%s.bin" % preName, "wb"))
    print(words)
     #读文件
    # read_file = open("../wordEmbeddingData/vector/vector_cbow_%s.bin"%preName, "rb")
    # load_file = pickle.load(read_file)
    # print('###################',len(load_file))
    # for i in load_file:
    #     print(i)


