import numpy as np
import data
import math
import random
import scipy
from scipy import optimize

def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)

def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels

# Load data
N_data, train_images, train_labels, test_images, test_labels = load_mnist()

N = 10000 # Number of data points in training set
train_images = train_images[:N,:] # 10k x 784 array
train_labels = train_labels[:N,:] # 10k x 10 array
train_images = np.ndarray.round(train_images) # Binarize the data

# Fit theta
Ncd = np.matmul(np.transpose(train_images),train_labels) # 784 x 10 array
Nc = train_labels.sum(axis=0)
#print(Nc)
#print(train_labels[0])
#print(train_labels[1])

#print("Ncd")
#print(Ncd)
#print("Nc")
#print(Nc)
thetaHat = (1+Ncd)/(2+Nc) # 784 x 10 array
save_images(np.transpose(thetaHat),'q1') # Plot thetaHat

logPtrain = np.matmul(train_images,np.log(thetaHat)) + \
np.matmul(1-train_images,np.log(1-thetaHat)) # 10k x 10 array
avLtrain = np.mean(np.sum(logPtrain*train_labels,axis=1))
logPtest = np.matmul(test_images,np.log(thetaHat)) + \
np.matmul(1-test_images,np.log(1-thetaHat)) # 10k x 10 array
avLtest = np.mean(np.sum(logPtest*test_labels,axis=1))

print(round(avLtrain,2), round(avLtest,2))
print("logpTrain") 
print(logPtrain)
print("npmeanptest")
print(np.mean(logPtest))
print("theta")
print(np.sum(thetaHat))

# Predictive accuracy
M = len(test_images) # Number of data points in test set
# 10k x 1 vector indicating whether a prediction was correct (1) or not (0):
accsTrain = train_labels[np.arange(N),logPtrain.argmax(1)]
accTrain = sum(accsTrain)/N
accsTest = test_labels[np.arange(M),logPtest.argmax(1)]
accTest = sum(accsTest)/M

print (round(accTrain*100,2),round(accTest*100,2))


c = (np.floor(np.random.rand(10)*10)).astype(int) # Pick the classes

xt = np.random.rand(10,784) # Prepare to sample 10 images
thresh = np.asmatrix(thetaHat[:,c].T) # Set thresholds
sample10 = 1*(thresh > np.asmatrix(xt)).T # Complete the sampling
save_images(np.transpose(sample10),'q2')



















