import numpy as np
import sys
from collections import Counter

class KNN:
    def __init__(self,k):
        self.k = k


def createdata(trainFile):
    file = open(trainFile)
    samples = []
    labels = []
    for line in file:
        d = line.strip().split(' ')
        tmp = []
        for i in range(len(d)):
            if i > 1:
                labels.append(d[i])
            else:
                tmp.append(float(d[i]))
        samples.append((tmp))
    file.close()
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels

def cal_distance(x,samples,num):
    tmp = []
    for j in range(num):
        ds = ((x[0]-samples[j][0])**2 + (x[1]-samples[j][1])**2)**0.5
        tmp.append(ds)
    return np.array(tmp)

def cal_result(sortedlist,labels,k):
    res = []
    n = len(sortedlist)-1
    for i in range(k):
        idex = sortedlist[i]
        res.append(labels[idex])
    #print(res)
    d = Counter(res)
    res = sorted(d.items(), key=lambda d: d[1], reverse=True)
    return res[0][0]


if __name__ == '__main__':
    if(len(sys.argv)!=4):
        print("Usage: python perceptron_duality.py k trainFile modelFile")
        exit(0)
    k = int(sys.argv[1])
    trainFile = sys.argv[2]
    testFile = sys.argv[3]
    samples, labels = createdata(trainFile)
    #print(samples.shape,labels.shape)
    x,_ = createdata(testFile)
    num = samples.shape[0]
    if k > num:
        k = num
    testnum = x.shape[0]
    tmp = np.zeros((num,))
    for i in range(testnum):
        tmp = cal_distance(x[i],samples,num)
        sortedlist = tmp.argsort()#<class 'numpy.ndarray'>
        res = cal_result(sortedlist,labels,k)
        print('测试数据'+str(x[i])+'的分类是：',res)




