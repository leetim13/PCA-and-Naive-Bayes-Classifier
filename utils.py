import numpy as np
from sklearn.decomposition import PCA #used to build intuition
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filename, load2=True, load3=True):
  """Loads data for 2's and 3's
  Inputs:
    filename: Name of the file.
    load2: If True, load data for 2's.
    load3: If True, load data for 3's.
  """
  assert (load2 or load3), "Atleast one dataset must be loaded."
  data = np.load(filename)
  if load2 and load3:
    inputs_train = np.hstack((data['train2'], data['train3']))
    inputs_valid = np.hstack((data['valid2'], data['valid3']))
    inputs_test = np.hstack((data['test2'], data['test3']))
    target_train = np.hstack((np.zeros((1, data['train2'].shape[1])), np.ones((1, data['train3'].shape[1]))))
    target_valid = np.hstack((np.zeros((1, data['valid2'].shape[1])), np.ones((1, data['valid3'].shape[1]))))
    target_test = np.hstack((np.zeros((1, data['test2'].shape[1])), np.ones((1, data['test3'].shape[1]))))
  else:
    if load2:
      inputs_train = data['train2']
      target_train = np.zeros((1, data['train2'].shape[1]))
      inputs_valid = data['valid2']
      target_valid = np.zeros((1, data['valid2'].shape[1]))
      inputs_test = data['test2']
      target_test = np.zeros((1, data['test2'].shape[1]))
    else:
      inputs_train = data['train3']
      target_train = np.zeros((1, data['train3'].shape[1]))
      inputs_valid = data['valid3']
      target_valid = np.zeros((1, data['valid3'].shape[1]))
      inputs_test = data['test3']
      target_test = np.zeros((1, data['test3'].shape[1]))

  return inputs_train.T, inputs_valid.T, inputs_test.T, target_train.T, target_valid.T, target_test.T

#Intution of displaying PCA compoenents vs images
def show(g, imshape, i, j, x, title=None):
        ax = fig.add_subplot(g[i, j], xticks=[], yticks=[])
        ax.imshow(x.reshape(imshape), interpolation='nearest')
        if title:
            ax.set_title(title, fontsize=12)
            
def plot_pca_components(x, coefficients=None, mean=0, components=None,
                        imshape=(16, 16), n_components=8, fontsize=12,
                        show_mean=True):
    '''Building intuition by viewing top k PCA
    '''
    if coefficients is None:
        coefficients = x
    if components is None:
        components = np.eye(len(coefficients), len(x))
    mean = np.zeros_like(x) + mean
    fig = plt.figure(figsize=(1.2 * (5 + n_components), 1.2 * 2))
    g = plt.GridSpec(2, 4 + bool(show_mean) + n_components, hspace=0.3)
    show(g, imshape, slice(2), slice(2), x, "True Original Image")
    approx = mean.copy()
    counter = 2
    if show_mean:
        show(g, imshape, 0, 2, np.zeros_like(x) + mean, r'$mean$')
        show(g, imshape,1, 2, approx)
        counter += 1
    for i in range(n_components):
        approx = approx + coefficients[i] * components[i]
        show(g, imshape,0, i + counter, components[i], r'$k_{0}$'.format(i + 1))
        show(g, imshape,1, i + counter, approx,
             r"${0:.2f} \cdot c_{1}$".format(coefficients[i], i + 1))
        if show_mean or i > 0:
            plt.gca().text(0, 1.05, '$+$', ha='right', va='bottom',
                           transform=plt.gca().transAxes, fontsize=fontsize)
    show(g, imshape, slice(2), slice(-2, None), approx, "Approx")
    return fig

def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(16, 16), #reshape into 16x16 in order to be displayed
                  cmap='binary', interpolation='nearest')

def first_k_components(training_data, k):
    '''
    Plot of first k_components vs eigen values
    '''
    mean = np.mean(training_data, axis =0)
    num_repeated = training_data.shape[0], 1
    centered_data = training_data - np.tile(mean, num_repeated) #subtract the mean of training data
    data_T = centered_data.T
    covariance_matrix = np.cov(data_T)
    eigen_values_cov , eigen_vectors_cov = np.linalg.eig(covariance_matrix )
    eigen_values_cov = eigen_values_cov[:: -1]
    eigen_vectors_cov = eigen_vectors_cov[:: -1]
    plt.figure()
    length = np.arange(0.0 , len(eigen_values_cov ) , 1) #[0,1,...,256]
    plt.plot(length, eigen_values_cov )
    plt.xlabel ("number of eigenvectors")
    plt.ylabel ("accuracy")
    plt.title ("plot of eigenvalues of covarience")
    plt.grid ( True )
    eigen_values = eigen_values_cov[:k]
    eigen_vectors =  eigen_vectors_cov[:k ,:]
#    print(eigen_values.shape) #(10,)
#    print(eigen_vectors.shape) #(10, 256)
    return eigen_values, eigen_vectors , mean

def one_nn_classifier(train_data, train_labels, valid_data, k=1) :
    N = len(valid_data)
    shape = N, 1
    valid_labels = np.zeros(shape) #initialize empty
    train_data_N = len(train_data)
    
    for i in range (N):
        min_index = -1
        min_value = np.inf
        for j in range (train_data_N):
            euclidean_distance = np.linalg.norm(valid_data[i]- train_data[j])
            if euclidean_distance < min_value :
                min_value = euclidean_distance
                min_index = j
        valid_labels[i] = train_labels[min_index]
    return valid_labels

def extract_eigen_features(training_data):
    mean = np.mean(training_data , axis =0)
    num_repeated = training_data.shape[0], 1
    centered_data = training_data - np.tile(mean, num_repeated)
    data_T = centered_data.T
    covariance_matrix = np.cov(data_T)
    eigen_values_cov , eigen_vectors_cov = np.linalg.eig(covariance_matrix )
    #    print(eigen_values_cov.shape) #(10,)
#    print(eigen_vectors_cov.shape) #(10, 256)
    sorted_eigen_values = eigen_values_cov.argsort()[:: -1] #vector of sorted eigen values asc
    eigen_values = eigen_values_cov[sorted_eigen_values]
    eigen_vectors = eigen_vectors_cov [:,sorted_eigen_values]
    return eigen_values , eigen_vectors , mean

def accuracy (prediction_value, target_value):
    return np.mean(target_value==prediction_value)

def train_model_pca(given_K , inputs_train , inputs_valid , target_train , target_valid):
    accuracy_list = []
    eigen_values , eigen_vectors , mean = extract_eigen_features(inputs_train)
    for k in given_K :
        code_vectors = eigen_vectors[: ,: k]
#        print(top_k_vector)
#        top_k_value = value [: k]
        num_repeated_training = (inputs_train.shape[0] , 1)
#        print(num_repeated_training.shape) #600
        centered_training_data = inputs_train - np.tile(mean, num_repeated_training )
        num_repeated_valid = (inputs_valid.shape[0] , 1)
#        print(num_repeated_valid.shape) #200
        centered_valid_data = inputs_valid - np .tile(mean, num_repeated_valid)
        
        #projection onto the low-dimensional space
        low_dim_space_valid = np.dot(centered_valid_data,  code_vectors)
        low_dim_space_train = np.dot(centered_training_data,  code_vectors)
        
        #using 1-NN classifier on K dimensional features
        low_dim_space_target = one_nn_classifier(low_dim_space_train , target_train , low_dim_space_valid )
        accuracy_ = accuracy(low_dim_space_target, target_valid)
        error = 1 - accuracy_
        accuracy_list.append(error)
    plt.figure()
    plt.grid(True)
    plt.plot(given_K , accuracy_list)
    plt.xlabel('first K principal components')
    plt.ylabel('Classification Error rates')
    plt.title ('plot of accuracy vs. #of eigen vectors')
    return accuracy_list


if __name__ == '__main__':
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = load_data("digits.npz")
#    print(inputs_train.shape)
#    print(inputs_valid.shape)
#    print(target_train.shape)
#    print(target_valid.shape)
    
    given_K = [2 , 5, 10 , 20 , 30]
#    view_eig_vector_images (10 , top_k_vector , mean )
    accuracy_list = train_model_pca(given_K, inputs_train , inputs_valid , target_train , target_valid)
    
#    print (accuracy_k)
    best_K = 20 #after selection
    accuracy_list = train_model_pca([best_K], inputs_train , inputs_test , target_train , target_test)
    print ("Error of K=" + str(best_K) + " = " + str(accuracy_list[0]))

    pca = PCA().fit(inputs_train) #only used to build intuition 
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of PCA components')
    plt.ylabel('amount of variance explained')
    pca = PCA(n_components=20)
    Xproj = pca.fit_transform(inputs_train)
    fig = plot_pca_components(inputs_train[155], Xproj[155],
                          pca.mean_, pca.components_)
    plot_digits(inputs_train)
    plt.show
    
    
    
    
    