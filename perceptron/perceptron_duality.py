import numpy as np
import matplotlib.pyplot as plt
import sys


def createdata(trainFile):
    file = open(trainFile)
    samples = []
    labels = []
    for line in file:
        d = line.strip().split(' ')
        tmp = []
        for i in range(len(d) - 1):
            tmp.append(int(d[i]))
        samples.append((tmp))
        labels.append([int(d[2])])
    file.close()
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels


class PerceptronD:
    def __init__(self, x, y, a=1):
        self.samples = x
        self.labels = y
        self.beta = a
        self.w = np.zeros((self.samples.shape[0], 1))
        self.b = 0
        self.numsamples = self.samples.shape[0]
        self.numfeatures = self.samples.shape[1]
        self.gram = self.gram()

    def gram(self):
        gram = np.zeros((self.numsamples, self.numsamples))
        for i in range(self.numsamples):
            for j in range(self.numsamples):
                gram[i][j] = np.dot(self.samples[i], self.samples[j])
        return gram

    def sign(self, i):
        h = self.w * self.labels
        y = np.dot(h.flatten(), self.gram[:,i]) + self.b
        return y

    def update(self, i):
        self.w = self.w + self.beta
        self.b = self.b + self.beta * self.labels[i]

    def saveModel(self, modelFile,weights):
        file = open(modelFile, 'w')
        tmp = ''
        for i in weights:
            tmp += str(i) + ' '
        tmp.strip()
        file.write('W: ' + tmp + '\n')
        file.write('b: ' + str(self.b[0]) + '\n')
        file.write('beta: ' + str(self.beta) + '\n')
        file.close()

    def cal_W(self):
        W = np.dot((self.w * self.labels).flatten(1), self.samples)
        return W
    def train(self):
        isFind = False
        while not isFind:
            count = 0  # 此次循环记录误分类点个数，没有则训练结束
            for i in range(self.numsamples):
                y_i = self.sign(i)
                if (self.labels[i] * y_i <= 0):
                    print('误分类点为：', self.samples[i], '此时的w和b分别是：', self.w, self.b)
                    self.update(i)
                    count += 1
            if (count == 0):
                print('最终训练得到的w和b分别是：', self.cal_W(), self.b)
                isFind = True
            weights = self.cal_W()
        return weights, self.b


class Picture:
    def __init__(self, samples, w, b):
        self.w = w
        self.b = b
        plt.figure(1)
        plt.title('Perceptron Learning Algorithm', size=14)
        plt.xlabel('x0-axis', size=10)
        plt.ylabel('x1-axis', size=10)
        x = np.linspace(0, 5, 100)
        y = (-self.b - self.w[0] * x) / self.w[1]
        plt.plot(x, y, color='r', label='sample data')
        for i in range(len(samples)):
            plt.scatter(samples[i][0], samples[i][1], s=50)
        plt.savefig('2d.png', dpi=75)

    def show(self):
        plt.show()


if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print("Usage: python perceptron_duality.py n trainFile modelFile")
        exit(0)
    a = int(sys.argv[1])
    trainFile = sys.argv[2]
    modelFile = sys.argv[3]
    samples, labels = createdata(trainFile)
    myperceptron = PerceptronD(samples, labels, a)
    weights, bias = myperceptron.train()
    myperceptron.saveModel(modelFile,weights)
    pic = Picture(samples, weights, bias)
    pic.show()
