from loadMNIST import *
from scipy.special import logsumexp 
import autograd as ag
import time
# np.random.seed(1)


COLORS = ["indianred", "palegoldenrod", "black", "gray"]


def get_images_by_label(images, labels, query_label):
		"""
		Helper function to return all images in the provided array which match the query label.
		"""
		assert images.shape[0] == labels.shape[0]
		matching_indices = labels == query_label
		return images[matching_indices]


class NaiveBayes:
    
    def avg_log_likelihood(self, X, y, theta):
        ll = 0
        for c in range(10):
            X_c = get_images_by_label(X, y, c)
            log_p_x = logsumexp(np.log(0.1) + np.dot(X_c, np.log(theta.T)) + np.dot((1. - X_c), np.log(1. - theta.T)), axis=1)
            ll += np.sum(np.dot(X_c, np.log(theta[c])) + np.dot((1. - X_c), np.log(1. - theta[c])) + np.log(0.1) - log_p_x)
        return ll / X.shape[0]
    
    def log_likelihood(self, X, y, theta):
#        print()
        ll = np.zeros((X.shape[0], 10))
#        print(ll.shape)
        log_p_x = logsumexp(np.log(0.1) + np.dot(X, np.log(theta.T)) + np.dot((1. - X), np.log(1. - theta.T)), axis=1)
#        print("log_p_x")
#        print(log_p_x.shape)
#        print("log")
#        print(np.log(0.1).shape)
        for c in range(10):
            ll[:, c] = np.dot(X, np.log(theta[c])) #+ np.dot((1. - X), np.log(1. - theta[c])) + np.log(0.1) - log_p_x
        return ll
    
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        
    def predict(self, X, y, theta, train=False, test=False):
        ll = self.log_likelihood(X, y, theta)
        pred = np.argmax(ll, axis=1)
        avg_ll = self.avg_log_likelihood(X, y, theta)
        accuracy = np.mean(pred == y)
        name = "test" if test else "train"
        print("average log-likelihood of naive bayes model on the {} set: ".format(name) + str(avg_ll))
#		print("accuracy of naive bayes model on the {} set: ".format(name) + str(accuracy))   
        
    def map_naive_bayes(self, plot=False):
        theta = np.zeros((10, 784))
        for c in range(10):
            images = get_images_by_label(self.train_images, self.train_labels, c)
            theta[c] = np.divide(np.sum(images, axis=0) + 1., images.shape[0] + 2.)
#		if plot:
#			save_images(theta, "theta_map.png")
        return theta









class GenerativeNaiveBayes:
    def __init__(self, theta):
        self.theta = theta
    

        
    def sample_plot(self):
#        c= np.random.multinomial(10, [0.1]*10)
#        c = np.random.choice(9,10)
#        images = np.zeros((10,784))
#        count = 0
#        for i in range(10):
#            for j in range((c[i])):
#                images[count] = np.random.binomial(1, self.theta[i]).reshape((1, 784))
#                count +=1
#        save_images(images, "samples.png")
        
        c = np.random.choice(9,10)
        images = np.zeros((10,784))
        count = 0
        print(self.theta.shape)
        for i in range(10):
            images[count] = np.random.binomial(1, self.theta[i]).reshape((1, 784))
            count +=1
        save_images(images, "samples.png")
#        c = (np.floor(np.random.rand(10)*10)).astype(int) # Pick the classes
#        xt = np.random.rand(10,784) # Prepare to sample 10 images
#        thresh = np.asmatrix(self.theta[:,c].T) # Set thresholds
#        sample10 = 1*(thresh > np.asmatrix(xt)).T # Complete the sampling
#        data.save_images(np.transpose(sample10),'ques2c')
#	def sample_plot(self):
#		"""
#		randomly sample and plot 10 binary images from the marginal distribution, p(x|theta, pi)
#		"""
#        
#		c = np.random.multinomial(10, [0.1]*10)
##        images = np.zeros((10, 784))
#        images = np.zeros((10,784))
#		for i in range(10):
#            for j in range((c[i])):
#				images[count] = np.random.binomial(1, self.theta[i]).reshape((1, 784))
#				count += 1
#		save_images(images, "samples.png")

#	def predict_half(self, X_top):
#		"""
#		plot the top half the image concatenated with the marginal distribution over each pixel in the bottom half.
#		"""
#		X_bot = np.zeros((X_top.shape[0], X_top.shape[1]))
#		theta_top, theta_bot = self.theta[:, :392].T, self.theta[:, 392:].T
#		for i in range(392):
#			constant = np.dot(X_top, np.log(theta_top)) + np.dot(1 - X_top, np.log(1 - theta_top))
#			X_bot[:, i] = logsumexp(np.add(constant, np.log(theta_bot[i])), axis=1) - logsumexp(constant, axis=1) 
#		save_images(np.concatenate((X_top, np.exp(X_bot)), axis=1), "predict_half.png")



if __name__ == '__main__':
	start = time.time()
	print("loading data...")
	N_data, train_images, train_labels, test_images, test_labels = load_mnist()
	train_labels = np.argmax(train_labels, axis=1)
	test_labels = np.argmax(test_labels, axis=1)

	print("trainning a Naive Bayes model...")
	nb_model = NaiveBayes(train_images, train_labels)
	theta_map = nb_model.map_naive_bayes(plot=True)
	nb_model.predict(train_images, train_labels, theta_map, train=True)
	nb_model.predict(test_images, test_labels, theta_map, test=True)

	print("training a generative Naive Bayes model...")
	gnb = GenerativeNaiveBayes(theta_map)
	gnb.sample_plot()
    
#	gnb.predict_half(train_images[:20,:392])

#	print("training a softmax model...")
#	lr_model = LogisticRegression(train_images, train_labels)
#	lr_model.predict(train_images, train_labels, train=True)
#	lr_model.predict(test_images, test_labels, test=True)
#
#	print("training K mean and GMM algorithms...")
#	initials = {'Nk': 200,
#				'MIU1': np.array([0.1, 0.1]),
#				'MIU2': np.array([6., 0.1]),
#				'COV': np.array([[10., 7.], [7., 10.]]),
#				'MIU1_HAT': np.array([0., 0.]),
#				'MIU2_HAT': np.array([1., 1.])
#				}
#	# Sampling data from a multivariate guassian distribution
#	c1 = np.random.multivariate_normal(initials['MIU1'], initials['COV'], initials['Nk'])
#	c2 = np.random.multivariate_normal(initials['MIU2'], initials['COV'], initials['Nk'])
#	kmean = KMean(initials, c1, c2)
#	kmean.plot_clusters()
#	kmean.train()
#	gmm = GaussianMixtures(initials, c1, c2)
#	gmm.train()
#	end = time.time()
#	print("running time: {}s".format(round(end - start, 2)))
	plt.show()


