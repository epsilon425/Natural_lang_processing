#!/usr/bin/python
import os
import sys
import glob
import math
import string
from collections import Counter
import operator
from collections import OrderedDict

indicator = 0
# ========== STOP WORDS ==========

def getStopWords(fin):
    return set([term.strip() for term in fin])

def removeStopWords(d, stopwords):
    for term in set(d.keys()):
        if term in stopwords:
            del d[term]

# ========== FREQUENCIES ==========

def getTermFrequencies(fin):
    tf = Counter()
    for line in fin: tf.update(map(lambda x: x.lower(), line.split()))
    return tf

def getDocumentFrequencies(tf):
    df = Counter()
    for d in tf: df.update(d.keys())
    return df

def getDocumentFrequency(df, term):
    if term in df: return df[term] + 1
    else: return 1

# ========== TF-IDF ==========

def getTFIDF(tf, df, dc):
    return tf * math.log(float(dc) / df)

def getTFIDFs(tf, df, dc):
    return [{k:getTFIDF(v, getDocumentFrequency(df,k), dc) for (k,v) in d.items()} for d in tf]

# ========== MEASUREMENTS ==========

def euclidean(d1, d2):
    s1 = set(d1.keys())
    s2 = set(d2.keys())
    t  = sum([(d1[k] - d2[k])**2 for k in s1.intersection(s2)])
    t += sum([d1[k]**2 for k in s1 - s2])
    t += sum([d2[k]**2 for k in s2 - s1])
    return math.sqrt(t)

def cosine(d1, d2):
    s1 = set(d1.keys())
    s2 = set(d2.keys())

    intersection = set(s1&s2)
    nume  = sum([(d1[k]*d2[k]) for k in intersection])
    sum1 = sum([d1[k]**2 for k in s1])
    sum2 = sum([d2[k]**2 for k in s2])
    denom = math.sqrt(sum1)*math.sqrt(sum2)
    return 1-(nume/denom)



# ========== EVALUATE ==========
def knn(trnFiles, devFiles, trnInsts, devInsts, sim, k, flag):

    if((indicator == 1) & (flag == "bow-euclidean")):
        fout = open('bow-euclidean-stopwords.txt','w')
    elif((indicator == 1) & (flag == "tfidf-euclidean")):
        fout = open('tfidf-euclidean-stopwords.txt','w')
    elif((indicator == 1) & (flag == "bow-cosine")):
        fout = open('bow-cosine-stopwords.txt','w')
    elif((indicator == 1) & (flag == "tfidf-cosine")):
        fout = open('tfidf-cosine-stopwords.txt','w')

    elif(flag=="bow-euclidean"):
        fout = open('bow-euclidean.txt','w')
    elif(flag == "tfidf-euclidean"):
        fout = open('tfidf-euclidean.txt','w')
    elif(flag == "bow-cosine"):
        fout = open('bow-cosine.txt','w')
    elif(flag == "tfidf-cosine"):
        fout = open('tfidf-cosine.txt','w')


    l = []
    correct = 0
    for t in range(len(devFiles)):
        neighbors = {}
        lastT = devFiles[t].split('_')[0]
        distances = {}
        votes = {}
        for t2 in range(len(trnFiles)):
            dist = sim(devInsts[t],trnInsts[t2])
            distances[trnFiles[t2]]= dist
        distances = OrderedDict(sorted(distances.items(), key=lambda t: t[1]))
        i = 0

        while (i!=k):
            x = distances.popitem(last=False)
            neighbors[x[0]] = x[1]
            i +=1
        for key in neighbors:
            split = key.split('_')[0]
            if split in votes:
                votes[split] +=1
            else:
                votes[split] = 1

        maX = max(votes.iteritems(), key = operator.itemgetter(1))[0]
        l.append([devFiles[t],maX])
        if (lastT==maX):
            correct+=1

    
    acc = 100.0 * correct / len(devFiles)
    print '%30s: %5.2f (%d/%d)' % (flag, acc, correct, len(devFiles))
    for key in l:
        fout.write(key[0]+' '+key[1])
        fout.write('\n')


# ===================================================
TRN_DIR = sys.argv[1]
DEV_DIR = sys.argv[2]
SW_FILE = sys.argv[3]
K = int(sys.argv[4])

print 'Read training data:'
trnFiles = sorted(glob.glob(os.path.join(TRN_DIR,'*.txt')))
trnTF    = [getTermFrequencies(open(filename)) for filename in trnFiles]
trnDF    = getDocumentFrequencies(trnTF)
trnDC    = len(trnFiles) + 1
trnTFIDF = getTFIDFs(trnTF, trnDF, trnDC)
print '- # of documents : %d' % len(trnTF)
print '- # of term types: %d' % len(trnDF)

print '\nRead development data:'
devFiles = sorted(glob.glob(os.path.join(DEV_DIR,'*.txt')))
devTF = [getTermFrequencies(open(filename)) for filename in devFiles]
devTFIDF = getTFIDFs(devTF, trnDF, trnDC)
print '- # of documents : %d' % len(devTF)

print '\nRead stopwords:'
sw = getStopWords(open(SW_FILE))
print '- # of stopwords : %d' % len(sw)

print '\nEvaluate including stopwords'
trnFiles = map(os.path.basename, trnFiles)
devFiles = map(os.path.basename, devFiles)

knn(trnFiles, devFiles, trnTF   , devTF   , euclidean, K,   'bow-euclidean')
knn(trnFiles, devFiles, trnTFIDF, devTFIDF, euclidean, K, 'tfidf-euclidean')
knn(trnFiles, devFiles, trnTF   , devTF   , cosine   , K,      'bow-cosine')
knn(trnFiles, devFiles, trnTFIDF, devTFIDF, cosine   , K,    'tfidf-cosine')

print '\nEvaluate excluding stopwords'
for d in trnTF   : removeStopWords(d, sw)
for d in trnTFIDF: removeStopWords(d, sw)
for d in devTF   : removeStopWords(d, sw)
for d in devTFIDF: removeStopWords(d, sw)
removeStopWords(trnDF, sw)
indicator = 1
knn(trnFiles, devFiles, trnTF   , devTF   , euclidean, K,   'bow-euclidean')
knn(trnFiles, devFiles, trnTFIDF, devTFIDF, euclidean, K, 'tfidf-euclidean')
knn(trnFiles, devFiles, trnTF   , devTF   , cosine   , K,      'bow-cosine')
knn(trnFiles, devFiles, trnTFIDF, devTFIDF, cosine   , K,    'tfidf-cosine')
