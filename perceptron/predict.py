import numpy as np
import sys


w = []
b = 0
beta = 0

def cal(item):
	global w,b
	res = 0
	for i in range(2):
		res += item[i] * w[i]
	res += b
	return res

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python predict.py testFile modelFile outFile")
        exit(0)
    testFile = open(sys.argv[1])
    modelFile = open(sys.argv[2])
    outFile = open(sys.argv[3],'w')
    line1 = modelFile.readline().strip()
    line2 = modelFile.readline().strip()
    line3 = modelFile.readline().strip()
    chunk = list(line1.split(' '))
    for key in range(1,len(chunk)):
        w.append(float(chunk[key]))
    b = float(list(line2.split(' '))[1])
    beta = int(list(line3.split(' '))[1])
    for line in testFile:
        chunk = list(line.strip().split(' '))
        tmp = []
        for i in range(2):
            tmp.append(int(chunk[i]))
        re = cal(tmp)
        if re > 0:
            outFile.write('+1\n')
        else:
            outFile.write('-1\n')
    outFile.close()
    testFile.close()
