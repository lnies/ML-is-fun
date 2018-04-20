from .config import *
from . import *


def calc_area(x, v):
    """
    Calculates the area beneath a spectrum
    """
    width = (x[len(x)-1]-x[0])/(len(x)-1)
    area = 0
    for i in range(len(x)):
        area += v[i] * width
    return area

def build_library(N, n, noise, random_states, Set, verbosity, poisson = POISSONCNTS):
    """
    Function wrapped around gen_set() for building the training library. 
        - If N is large, generates multiple msgpack files and then zips them.
    """
    print("Poissoncnts (build): %f" %(poisson))
    start = time.time()
    label_counter = 1
    if ( N > 1e5 ):
        # Create files with size of 1e5 until all spectra are created
        # Check if file already exists
        if (os.path.isfile("./data/"+Set+".tar.bz2") == True):
            sys.stderr.write('Error: File already exists! \n')
            return 0
        tarfile.open(name = "./data/"+Set+".tar.bz2", mode = "x:bz2")
        tar = tarfile.open(name = "./data/"+Set+".tar.bz2", mode = "w:bz2")
        while ( N >= 1e5 ):
            gen_set(int(1e5), n, noise, random_states, "temp_"+str(label_counter)+".sublibrary", verbosity, poissoncnts = poisson)
            # Zip files to one compressed container and delete single them afterwards
            tar.add("./data/temp_"+str(label_counter)+".sublibrary.")
            # Remove temporary file
            os.remove("./data/temp_"+str(label_counter)+".sublibrary.")
            N -= 1e5
            label_counter += 1
        tar.close()
        end = time.time()
        if (verbosity > 0):
            print("+++++++++++++")
            print("Time for generating library: %3.2fs" % (end-start))
            print("+++++++++++++")
    else:
        if (os.path.isfile("./data/"+Set+".tar.bz2") == True):
            sys.stderr.write('Error: File already exists! \n')
            return 0
        tarfile.open(name = "./data/"+Set+".tar.bz2", mode = "x:bz2")
        tar = tarfile.open(name = "./data/"+Set+".tar.bz2", mode = "w:bz2")
        gen_set(int(N), n, noise, random_states, "temp_"+str(label_counter)+".sublibrary", verbosity, poissoncnts = poisson)
        tar.add("./data/temp_"+str(label_counter)+".sublibrary.")
        os.remove("./data/temp_"+str(label_counter)+".sublibrary.")
        end = time.time()
        if (verbosity > 0):
            print("+++++++++++++")
            print("Time for generating library: %3.2fs" % (end-start))
            print("+++++++++++++")
    tar.close()

def gen_set(N, n, noise, random_states, Set, verbosity, poissoncnts = POISSONCNTS): 
    """
    - Generates a set with N spectra by using the superposition of TWO Voigt profiles with randomly choosen
        parameters 
            gamma1: HWHM of Lorentzian part of Voigt profile 1 
            gamma2: HWHM of Lorentzian part of Voigt profile 2
            sigma1: Standard uncertainty of Gaussian part of Voigt profile 1
            sigma2: Standard uncertainty of Gaussian part of Voigt profile 2
            epsilons: offset to energy E, dE, Ratios
        The Energy E (K alpha1) is centered around 2014eV 
        Splitting is set to 0.85eV +- 0.05
        Ratios are set to 1.7 pm 0.5
    """
    print("Poissoncnts (gen): %f" %(poissoncnts))
    assert n <= MAXNUMSTATES, "n is greater than MAXNUMSTATES. Check config"
    if (verbosity > 0):
        start = time.time()
    # Creating the empty data dictionary to store data
    X = {
        'states': np.zeros(N),
        'energies': np.zeros(N*MAXNUMSTATES).reshape(N,MAXNUMSTATES),
        'energies_binned': np.zeros(N*MAXNUMSTATES).reshape(N,MAXNUMSTATES),
        'splittings': np.zeros(N*MAXNUMSTATES).reshape(N,MAXNUMSTATES),
        'ratios': np.zeros(N*MAXNUMSTATES).reshape(N,MAXNUMSTATES),
        'features': np.zeros(N*len(GRID_FEATURES)).reshape(N,len(GRID_FEATURES)),
        'plot': np.zeros(N*len(GRID_PLOT)).reshape(N,len(GRID_PLOT)), 
        'energy_label': np.zeros(N)
    }
    # Array for storing runtime 
    runtime = np.array([])
    # If random_state is true, pick random number of oxidation states in range n
    """
    For loop loops N times to create N spectra. The single spectrum is evaluate and fitted
    on range x to get equal x values as features (Note: When trained on grid defined by x then
    real data must also be sampled on same grid!). File format:
        File dimensions: N x (2 + d), where d is number of grid points resulting from grid x
        [E dE x1 x2 ... xd]
    """
    if (verbosity > 0):
        loop_time_start = time.time()
    for i in range(N):
        # Create empty feature array
        feature_array = np.zeros(len(GRID_FEATURES))
        feature_plot = np.zeros(len(GRID_PLOT))
        ## Generate superposition of n different oxidation states 
        # Save number of states in dictionary
        # If random_state is true, pick random number of oxidation states in range n
        if (random_states == True):
            k = np.random.randint(1,n+1)
        else:
            k = n
        X["states"][i] = k
        for j in range(k):
            # Generate random distribution (+- 1) around central value of energie E
            E_epsilon = (np.random.random_sample()-0.5)*2
            # Generate random distribution (+- 0.1) around central value of energie dE
            dE_epsilon = (np.random.random_sample()-0.5)*(0.4)
            # Generate random distribution (+- 0.05) around central value of amplitude L1
            dL1 = (np.random.random_sample()-0.5)/15
            # Generate random distribution (+- 0.05) around central value of amplitude L2
            dL2 = (np.random.random_sample()-0.5)/15
            l1 = L1 + dL1
            l2 = L2 + dL2
            E = MEAN_ENERGY + E_epsilon
            dE = MEAN_ENERGY_SHIFT + dE_epsilon
            v1 = Voigt1D(x_0=E-dE, amplitude_L=l1, fwhm_L=2*GAMMA1, fwhm_G=2*SIGMA1*np.sqrt(2*np.log(2)))
            v2 = Voigt1D(x_0=E, amplitude_L=l2, fwhm_L=2*GAMMA2, fwhm_G=2*SIGMA2*np.sqrt(2*np.log(2)))
            # Superpose the states
            feature_plot += v1(GRID_PLOT)+v2(GRID_PLOT)
            feature_array += v1(GRID_FEATURES)+v2(GRID_FEATURES)
            # Calculate the ratio of the areas
            R = calc_area(GRID_FEATURES, v2(GRID_FEATURES))/calc_area(GRID_FEATURES,v1(GRID_FEATURES))
            ### Discretize Energies for classification option
            # Define energy bins
            #bins = np.arange(0,2018,0.01)
            bins = np.arange(-5,2025,ENERGY_BIN_WIDTH_TARGET)
            # Apply this grid on enegry and center energy to the bin center to increase precision
            binned = np.digitize(E, bins, right=True)
            E_binned = bins[binned]#-0.005
            # Save values in dictionary
            X["energies"][i,j+0] = E
            X["energies_binned"][i,j] = E_binned
            X["splittings"][i,j] = dE
            X["ratios"][i,j] = R
        # Choose randomly one energy which will be treated as the label
        X["energy_label"][i] = random.choice(X['energies_binned'][i][X['energies_binned'][0]!=0])
        ## Apply noise to data    
        if (noise == True):
            # Normalize spectrum to 1
            amp_feature = np.amax(feature_array)
            amp_plot = np.max(feature_plot)
            feature_array  /= amp_feature
            feature_plot /= amp_plot
            # Apply poisson noise to the data, magnify amplitudes to get poisson function working
            feature_array = np.multiply(feature_array, poissoncnts)
            feature_array = np.random.poisson(feature_array)
            feature_plot = np.multiply(feature_plot, poissoncnts)
            feature_plot = np.random.poisson(feature_plot)
            # Scale down again to 1
            feature_array = np.divide(feature_array, poissoncnts)
            feature_plot = np.divide(feature_plot, poissoncnts)
            # Fill ddictionary:
            X["features"][i] = feature_array
            X["plot"][i] = feature_plot
        else: 
            # Normalize spectrum to 1
            amp_feature = np.amax(feature_array)
            amp_plot = np.max(feature_plot)
            feature_array  /= amp_feature
            feature_plot /= amp_plot
            # Fill ddictionary:
            X["features"][i] = feature_array
            X["plot"][i] = feature_plot
        # Runtime control
        if (verbosity > 0):
            if ( i % (N/10) == 0 ):
                loop_time_end = time.time()
                time_diff = loop_time_end-loop_time_start
                runtime = np.append(runtime, time_diff)
                print("Progress: %i/%i, time for loop: %3.2fs" % (i , N, time_diff))
                loop_time_start = time.time()
    # Save dictinoary as library in .msgpack file
    with open('./data/'+Set,'wb') as f:
        dill.dump(X,f)
    # Some verbosity output
    if (verbosity > 0):
        end = time.time()
        print("Time for generating the "+Set+" set:", end-start)
        if (verbosity > 1):
            plt.figure()
            plt.title("Runtime for generating the data set "+Set)
            plt.plot(runtime, label="Runtime per N/10 loops")
            plt.grid(True)
            plt.legend()
            plotly_show()
    return X

def load_library(Set, fraction=100, verbosity=False):
    """
    Function wrapped around read_set() for loading the training library. 
    """
    start = time.time()
    tar = tarfile.open("./data/"+Set+".tar.bz2", "r:bz2")
    label_counter = 1
    k = 0
    N = 0
    l = 0
    X = {
        'states': np.zeros(N),
        'energies': np.zeros(N*k).reshape(N,k),
        'energies_binned': np.zeros(N*k).reshape(N,k),
        'splittings': np.zeros(N*k).reshape(N,k),
        'ratios': np.zeros(N*k).reshape(N,k),
        'features': np.zeros(N*l).reshape(N,l),
        'plot': np.zeros(N*l).reshape(N,l),
        'energy_label': np.zeros(N)
    }
    for i in (tar):
        if ( label_counter > fraction ):
            break
        start_loop = time.time()
        tar.extract(i)
        temp = read_set("temp_"+str(label_counter)+".sublibrary", verbosity)
        N += len(temp['states'])
        for item in legend:
            if (item == 'states' or item == 'energy_label'):
                X[item] = np.append(X[item], temp[item])
            else:
                d = len(temp[item][0])
                X[item] = np.append(X[item], temp[item]).reshape(N,d)
        end_loop = time.time()
        if (verbosity > 0):
            print("Time for unpacking: %3.2fs" % (end_loop-start_loop))
        os.remove("./data/temp_"+str(label_counter)+".sublibrary")
        label_counter += 1
    tar.close()
    end = time.time()
    if (verbosity > 0):
        print("+++++++++++")
        print("Time for loading the library: %3.2fs" % (end-start))
        print("+++++++++++")
    return X

def read_set(Set, verbosity=False):
    """ 
    Read data and store it in (Nxd) Martix, where N donates 
    the observation (single spectrum) and d the dth feature 
    (datapoint given by choosing x). The data gets fitted 
    by the Splines fit. Also, noise is added when reading the
    data if flag is set.
    """
    start = time.time()
    with open('./data/'+Set,'rb') as f:
        X = dill.load(f)
    end = time.time()
    if (verbosity > 0):
        print("Time for reading "+Set+" set: %3.2fs" % (end-start))
    return X

def scale_input(X):
    """
    Feature skaling for NN apporach. It is "highly recommended" to scale input data to either [0:1] or [-1:+1] 
    or standardize it to have mean 0 and variance 1
    Source:
    http://scikit-learn.org/stable/modules/neural_networks_supervised.html#regression
    This function standardizes X 
    """
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    N = len(X)
    d = len(X[0]+n_labels)
    scaler.fit(X)  
    X = scaler.transform(X) 
    #for i in range(len(X)):
    #    plt.plot(x,X[i])
    #plt.grid(True)
    #plotly_show()
    return X

def NN_train(Xtrain, Ytrain, Xdev, Ydev, model, verbosity):
    """
    Trains given model on data X and labels y. Returns trainings score and instance of trained model
    """
    start = time.time()
    # Create target classes (energies only expected between 2013eV and 2015eV)
    grid_target_int = GRID_TARGET*100
    grid_target_int = grid_target_int.astype(int)
    # Create error array to store mean squared error
    train_error = np.array([])
    dev_error = np.array([])
    # Set out pipe to catch stdout for getting verbosity output of model.fit
    if (verbosity > 0):
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
    # Perform training 
    for i in range(model.max_iter):
        model.partial_fit(Xtrain, Ytrain, classes=np.unique(grid_target_int))
        # Calculate errors
        predict_train = model.predict(Xtrain)/100
        predict_dev = model.predict(Xdev)/100
        train_error = np.append(train_error, np.sum(np.absolute(Ytrain/100-predict_train)))
        dev_error = np.append(dev_error, 8*np.sum(np.absolute(Ydev/100-predict_dev)))# factor 8 because dev set is 8 times smaller than the train set
    # Delete pipe
    if (verbosity > 0):
        sys.stdout = old_stdout
    # Save verbosity output (training loss) in variable
    loss = np.array(0)
    loss = np.delete(loss, 0)
    if (verbosity > 0):
        verbosity_output = mystdout.getvalue()
        verbosity_output = np.array(verbosity_output.split(' '))
        for i in range(4,len(verbosity_output),4):
            if (verbosity_output[i].split('\n')[0] == 'improve'):
                break
            else:
                loss = np.append(loss, float(verbosity_output[i].split('\n')[0]))
    # Save score of training
    score = model.score(Xtrain,Ytrain)
    end = time.time()
    # Print training statistics depending onb verbosity level
    if (verbosity > 0):
        print("Training time: %3.2f " % (end - start))
        print("Training score: %3.2f " % (score))
        if (verbosity > 1):
            plt.figure()
            plt.title("Training loss per epoch")
            plt.semilogy(loss, label="Loss")
            plt.grid(True)
            plt.legend()
            plotly_show()       
            plt.figure()
            plt.title("Squared error per epoch")
            plt.semilogy(train_error, label="Training error")
            plt.semilogy(dev_error, label="Dev error")
            plt.grid(True)
            plt.legend()
            plotly_show()       
    return score

def test_envir(library, n, model, verbosity, scaling, pca, k):
    """
    Trains and tests a NN on a given label
    """
    Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest = split_library(library, "energy_label")
    Ytrain = Ytrain*100
    Ytrain_int = Ytrain.astype(int)
    Ytest = Ytest*100
    Ytest_int = Ytest.astype(int)
    Ydev = Ydev*100
    Ydev_int = Ydev.astype(int)
    scores = np.array([])
    predict_train = np.array([])
    predict_test = np.array([])
    predict_dev =np.array([])
    predict_proba_train = np.array([])
    predict_proba_test = np.array([])
    predict_proba_dev = np.array([])
    # IF PCA is true, perform PCA with following feature mapping
    if (pca == True):
        X_hat_train, X_rec_train = PCA(Xtrain, k)
        X_hat_dev, X_rec_dev = PCA(Xdev, k)
        X_hat_test, X_rec_test = PCA(Xtest, k)
        Xtrain, Xdev, Xtest = gen_pol_feat(X_hat_train, X_hat_dev, X_hat_test)
    # Start testing
    score = NN_train(Xtrain, Ytrain_int, Xdev, Ydev_int, model, verbosity)
    scores = np.append(scores, score)
    
    #Save model via
    joblib.dump(model, './data/'+str(1)+'_neural_network.pkl')
    #Load model via
    model2 = joblib.load('./data/'+str(1)+'_neural_network.pkl')
    
    predict_train = model.predict(Xtrain)/100
    predict_proba_train = model.predict_proba(Xtrain)
    predict_dev = model.predict(Xdev)/100
    predict_proba_dev = model.predict_proba(Xdev)
    plt.plot(Ytrain/100-predict_train, label=("Train: Loss energy on energy "+str(1)))
    plt.plot(Ydev/100-predict_dev, label=("Dev: Loss energy on energy "+str(1)))
    abserr_train = np.absolute(Ytrain/100-predict_train)
    abserr_train = np.sum(abserr_train)
    abserr_dev = np.absolute(Ydev/100-predict_dev)
    abserr_dev = np.sum(abserr_dev)
    print("Mean error per prediction in training set: %3.2f on energy label" % (abserr_train/len(Xtrain)))
    print("Mean error per prediction in dev set: %3.2f on energy label" % (abserr_dev/len(Xdev)))
    plt.xlabel("datapoint")
    plt.ylabel("Error in arb. units")
    plt.title("Error on true label")
    plt.legend()
    plt.grid(True)
    plotly_show()
    return predict_train, predict_dev, predict_proba_train, predict_proba_dev

def split_library(library, key):
    """
    Function to split a library in training, dev, and test set after good 'ol ratio 80%:10%:10%
    with feature "key"
    """
    N = len(library["states"])
    N_train = int(N*0.8)
    N_dev = N_test = int(N*0.1)
    Xtrain = library["features"][0:N_train]
    Ytrain = library[key][0:N_train]
    Xdev = library["features"][N_train : N_train + N_dev]
    Ydev = library[key][N_train : N_train + N_dev]
    Xtest = library["features"][N_train + N_dev : N_train + 2*N_dev]
    Ytest = library[key][N_train + N_dev : N_train + 2*N_dev]
    return Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest

def calc_cov_mat(Xtrain):
    """
    Calculate the covariance matrix of a matrix based on matrix algebra
    """
    X = Xtrain
    n = len(X)
    d = len(X[0])
    # First center the matrix X by substracting the mean
    Xc = X - np.mean(X, axis = 0)
    # Then calculate the covariance matrix
    sigma = np.dot(np.transpose(Xc), Xc)
    sigma /= n
    return sigma

def eigen_analysis(Xtrain):
    """
    Calculate the Eigenvalues of the data matrix using SVD (single value decomposition)
    """
    sigma = calc_cov_mat(Xtrain)
    u, lamda, v = np.linalg.svd(sigma, full_matrices=1, compute_uv = 1)
    
    print("+++ Information for the calculation of eigenvalues +++")
    print("Eigenvalue Lambda1:", lamda[0])
    print("Eigenvalue Lambda2:", lamda[1])
    print("Eigenvalue Lambda10:", lamda[9])
    print("Eigenvalue Lambda30:", lamda[29])
    print("Eigenvalue Lambda50:", lamda[49])
    print("Eigenvalue Lambda100:", lamda[99])
    print("Eigenvalue Lambda200:", lamda[199])
    print("Sum of eigenvalues:", np.sum(lamda))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return u, lamda, v

def PCA(Xtrain, k) :
    """
    Reconstruct the data after projecting down to k dimensions
    """
    start = time.time()
    X = Xtrain
    n = len(X)
    d = len(X[0])
    # First center the matrix X by substracting the mean
    Xc = X - np.mean(X, axis = 0)
    # Apply SVD for getting eigenvalues and vectors
    u, lamda, v = eigen_analysis(Xtrain) 
    # Create eigenvector matrix of top k eigenvecors
    U_hat = v[0:k]
    U_hat = np.transpose(U_hat)
    # Apply the dimensionality reduction
    X_hat = np.dot(Xc, U_hat) 
    # Now reconstruct data
    # Add the mean back to the data
    X_rec = np.dot(X_hat,np.transpose(U_hat)) + np.mean(X, axis = 0)
    end = time.time()
    print("Reconstruction time %f3.2s" % (end-start))
    return X_hat, X_rec

def recon_error(Xtrain):
    """
    Plot the fractional reconstruction error for k = 100
    """
    d = np.size(Xtrain[0])
    u, lamda, v = eigen_analysis(Xtrain)
    lamda_sum = np.sum(lamda)
    data = np.array(0)
    data = np.delete(data, 0)
    for k in range(d):
        upstairs = 0
        for i in range(k):
            upstairs += lamda[i]
        error = 1 - ( upstairs ) / ( lamda_sum )
        data = np.append(data, error)
    x = np.arange(1,d+1)
    plt.figure()
    plt.plot(x, data, label="Fractional error")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Index k")
    plt.ylabel("Fractional reconstruction error")
    plotly_show()

def gen_pol_feat(Xtrain_small, Xdev_small, Xtest_small):
    ## Function taken from class canvas (post by Edith Heiter) at CSE 448 @ UW in W2018
    # Compute nxdxd matrices containing the outer product x^Tx for all n entries
    Xtrain_outer = np.einsum('ij, ik -> ijk', Xtrain_small, Xtrain_small, optimize=True)
    Xdev_outer = np.einsum('ij, ik -> ijk', Xdev_small, Xdev_small, optimize=True)
    Xtest_outer = np.einsum('ij, ik -> ijk', Xtest_small, Xtest_small, optimize=True)
    # Compute arrays of indices for upper triangle part of a dxd matrix
    j, k = np.triu_indices(Xtrain_small.shape[1])
    # Flatten the three dimensional array and keep only the upper triangle
    Xtrain = Xtrain_outer[:, j, k]
    Xdev = Xdev_outer[:, j, k]
    Xtest = Xtest_outer[:, j, k]
    return Xtrain, Xdev, Xtest













