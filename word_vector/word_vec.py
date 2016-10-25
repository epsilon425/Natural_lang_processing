#!/usr/bin/python
import sys
import numpy as np
 
fout = open('Result.txt','w')

def readVocab(fin):
    return ([word.strip() for word in fin])

def readWordVectors(fin, vocab):
    word_vectors = {}
    header = fin.readline()
    vocab_size, vector_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * vector_size
    print 'All vocab size: ', vocab_size
    print 'Our vocab size: ', len(vocab)
    print 'Vector size   : ', vector_size

    for line in xrange(vocab_size):
        word = []
        while True:
            ch = fin.read(1)
            if ch == b' ':
                word = ''.join(word)
                break
            if ch != b'\n':
                word.append(ch)

        if vocab and word in vocab: word_vectors[word] = np.fromstring(fin.read(binary_len), dtype='float32')
        else: fin.read(binary_len)

    return word_vectors

def getCosineSimilarity(v1, v2):
    num  = np.dot(v1, v2)
    den1 = np.sqrt(np.dot(v1, v1))
    den2 = np.sqrt(np.dot(v2, v2))
    return num / (den1 * den2)

def getSimilarities(wv, v1, wordcheck):
    text1 = wordcheck.split('-')
    l=[]
    for(w2, v2) in wv.items():
        text2 = w2.split('-')
        a = (text1[0] != text1[1] and text1[0] != text2[0] and text1[0] != text2[1] and text1[1] != text2[0] and text1[1] != text2[1] and text2[0] != text2[1])
        if(a):
            l.append((getCosineSimilarity(v1, v2), w2))
    return sorted(l, reverse=True)



VOCAB_FILE = sys.argv[1] # vocab.txt
W2V_FILE = sys.argv[2]   # w2v.bin
K = 5

vocab = readVocab(open(VOCAB_FILE))
wv = readWordVectors(open(W2V_FILE), vocab)
diff = {}

for vocab in open(VOCAB_FILE):
    a = vocab.strip()
    for vocab2 in open(VOCAB_FILE):
        b = vocab2.strip()
        if(b==a):
            continue
        diff[a+'-'+b] = wv[a] - wv[b]



for keys,values in diff.items():

    l = getSimilarities(diff,diff[keys], keys)
    for(v,w) in l[:K]:
        fout.write(str(keys+'='+w+'\n'))



"""print 'take:'
l = getSimilarities(wv, wv['take'])
for (v, w) in l[:K]: print '  ', w, v

print 'take - do + carry:'
l = getSimilarities(wv, wv['take'] - wv['do'] + wv['carry'])
for (v, w) in l[:K]: print '  ', w, v"""

