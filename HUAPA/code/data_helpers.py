#-*- coding: utf-8 -*-
#author: Zhen Wu

import numpy as np

def load_embedding(embedding_file_path, corpus, embedding_dim):
    wordset = set();
    for line in corpus:
        line = line.strip().split()
        for w in line:
            wordset.add(w.lower())
    words_dict = dict(); word_embedding = []; index = 1
    words_dict['$EOF$'] = 0  #add EOF
    word_embedding.append(np.zeros(embedding_dim))
    with open(embedding_file_path, 'r') as f:
        for line in f:
            check = line.strip().split()
            if len(check) == 2: continue
            line = line.strip().split()
            if line[0] not in wordset: continue
            embedding = [float(s) for s in line[1:]]
            word_embedding.append(embedding)
            words_dict[line[0]] = index
            index +=1
    return np.asarray(word_embedding), words_dict


def fit_transform(x_text, words_dict, max_sen_len, max_doc_len):
    x, sen_len, doc_len = [], [], []
    for index, doc in enumerate(x_text):
        t_sen_len = [0] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len), dtype=int)
        sentences = doc.split('<sssss>')
        i = 0
        for sen in sentences:
            j = 0
            for word in sen.strip().split():
                if j >= max_sen_len:
                    break
                if word not in words_dict: continue
                t_x[i, j] = words_dict[word]
                j += 1
            t_sen_len[i] = j
            i += 1
            if i >= max_doc_len:
                break
        doc_len.append(i)
        sen_len.append(t_sen_len)
        x.append(t_x)
    return np.asarray(x), np.asarray(sen_len), np.asarray(doc_len)

class Dataset(object):
    def __init__(self, data_file):
        self.t_usr = []
        self.t_prd = []
        self.t_label = []
        self.t_docs = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip().decode('utf8', 'ignore').split('\t\t')
                self.t_usr.append(line[0])
                self.t_prd.append(line[1])
                self.t_label.append(int(line[2])-1)
                self.t_docs.append(line[3].lower())
        self.data_size = len(self.t_docs)

    def get_usr_prd_dict(self):
        usrdict, prddict = dict(), dict()
        usridx, prdidx = 0, 0
        for u in self.t_usr:
            if u not in usrdict:
                usrdict[u] = usridx
                usridx += 1
        for p in self.t_prd:
            if p not in prddict:
                prddict[p] = prdidx
                prdidx += 1
        return usrdict, prddict

    def genBatch(self, usrdict, prddict, wordsdict, batch_size, max_sen_len, max_doc_len, n_class):
        self.epoch = len(self.t_docs) / batch_size
        if len(self.t_docs) % batch_size != 0:
            self.epoch += 1
        self.usr = []
        self.prd = []
        self.label = []
        self.docs = []
        self.sen_len = []
        self.doc_len = []

        for i in xrange(self.epoch):
            self.usr.append(np.asarray(map(lambda x: usrdict[x], self.t_usr[i*batch_size:(i+1)*batch_size]), dtype=np.int32))
            self.prd.append(np.asarray(map(lambda x: prddict[x], self.t_prd[i*batch_size:(i+1)*batch_size]), dtype=np.int32))
            self.label.append(np.eye(n_class, dtype=np.float32)[self.t_label[i*batch_size:(i+1)*batch_size]])
            b_docs, b_sen_len, b_doc_len = fit_transform(self.t_docs[i*batch_size:(i+1)*batch_size],
                                                         wordsdict, max_sen_len, max_doc_len)
            self.docs.append(b_docs)
            self.sen_len.append(b_sen_len)
            self.doc_len.append(b_doc_len)

    def batch_iter(self, usrdict, prddict, wordsdict, n_class, batch_size, num_epochs, max_sen_len, max_doc_len, shuffle=True):
        data_size = len(self.t_docs)
        num_batches_per_epoch = int(data_size / batch_size) + \
                                (1 if data_size % batch_size else 0)
        self.t_usr = np.asarray(self.t_usr)
        self.t_prd = np.asarray(self.t_prd)
        self.t_label = np.asarray(self.t_label)
        self.t_docs = np.asarray(self.t_docs)

        for epoch in xrange(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                self.t_usr = self.t_usr[shuffle_indices]
                self.t_prd = self.t_prd[shuffle_indices]
                self.t_label = self.t_label[shuffle_indices]
                self.t_docs = self.t_docs[shuffle_indices]

            for batch_num in xrange(num_batches_per_epoch):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, data_size)
                usr = map(lambda x: usrdict[x], self.t_usr[start:end])
                prd = map(lambda x: prddict[x], self.t_prd[start:end])
                label = np.eye(n_class, dtype=np.float32)[self.t_label[start:end]]
                docs, sen_len, doc_len = fit_transform(self.t_docs[start:end], wordsdict, max_sen_len, max_doc_len)
                batch_data = zip(usr, prd, docs, label, sen_len, doc_len)
                yield batch_data


