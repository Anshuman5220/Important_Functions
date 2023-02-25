import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler


#########################################
### set up the random undersampling class
#########################################

rus = RandomUnderSampler(
    sampling_strategy='auto',  # samples only the majority class
    random_state=0,  # for reproducibility
    replacement=True # if it should resample with replacement
)  
X_resampled, y_resampled = rus.fit_resample(X, y)

# now, I will resample the data, so that I obtain
# twice as many observations from the majority as
# those from the minority

rus = RandomUnderSampler(
    sampling_strategy= 0.5,  # remember balancing ratio = x min / x maj
    random_state=0,  
    replacement=False # if it should resample with replacement
)  

X_resampled, y_resampled = rus.fit_resample(X, y)

# and we can also specify how many observations we want
# from each class

rus = RandomUnderSampler(
    sampling_strategy= {0:100, 1:15},  # remember balancing ratio = x min / x maj
    random_state=0,  
    replacement=False # if it should resample with replacement
)  

X_resampled, y_resampled = rus.fit_resample(X, y)


# function to train random forests and evaluate the performance

def run_randomForests(X_train, X_test, y_train, y_test):
    
    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
    rf.fit(X_train, y_train)

    print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

run_randomForests(X_train,
                  X_test,
                  y_train,
                  y_test)

#########################################
### set up the Condensed-Nearest-Neighbours undersampling class
#########################################

from imblearn.under_sampling import CondensedNearestNeighbour

# set up condensed nearest neighbour transformer

cnn = CondensedNearestNeighbour(
    sampling_strategy='auto',  # undersamples only the majority class
    random_state=0,  # for reproducibility
    n_neighbors=1,# default
    n_jobs=4)  # I have 4 cores in my laptop

X_resampled, y_resampled = cnn.fit_resample(X, y)

## Plotting 
sns.scatterplot(
    data=X, x="varA", y="varB", hue=y
)

plt.title('Original dataset')
plt.show()

# plot undersampled data

sns.scatterplot(
    data=X_resampled, x="varA", y="varB", hue=y_resampled
)

plt.title('Undersampled dataset')
plt.show()

### Partially seperated class

# set up condensed nearest neighbour transformer

cnn = CondensedNearestNeighbour(
    sampling_strategy='auto',  # undersamples only the majority class
    random_state=0,  # for reproducibility
    n_neighbors=1,
    n_jobs=4)  # I have 4 cores in my laptop

X_resampled, y_resampled = cnn.fit_resample(X, y)

## Tomek Links

from imblearn.under_sampling import TomekLinks

# set up Tomek Links

tl = TomekLinks(
    sampling_strategy='auto',  # undersamples only the majority class
    n_jobs=4)  # I have 4 cores in my laptop

X_resampled, y_resampled = tl.fit_resample(X, y)


######################################################
### set up the One Sided Selection undersampling class
######################################################

from imblearn.under_sampling import OneSidedSelection

oss = OneSidedSelection(
    sampling_strategy='auto',  # undersamples only the majority class
    random_state=0,  # for reproducibility
    n_neighbors=1,# default, algo to find the hardest instances.
    n_jobs=4)  # I have 4 cores in my laptop

X_resampled, y_resampled = oss.fit_resample(X, y)


######################################################
### set up the Edited-Nearest-Neighbours undersampling class
######################################################

from imblearn.under_sampling import EditedNearestNeighbours
# set up edited nearest neighbour

enn = EditedNearestNeighbours(
    sampling_strategy='auto',  # undersamples only the majority class
    n_neighbors=3, # the number of neighbours to examine
    kind_sel='all',  # all neighbours need to have the same label as the observation examined
    n_jobs=4)  # I have 4 cores in my laptop

X_resampled, y_resampled = enn.fit_resample(X, y)

#####################################################################
### set up the Repeated-Edited-Nearest-Neighbours undersampling class
#####################################################################

from imblearn.under_sampling import RepeatedEditedNearestNeighbours
# set up repeated edited nearest neighbour

renn = RepeatedEditedNearestNeighbours(
    sampling_strategy='auto',# removes only the majority class
    n_neighbors=3, # the number of neighbours to examine
    kind_sel='all', # all neighbouring observations should show the same class
    n_jobs=4, # 4 processors in my laptop
    max_iter=100) # maximum number of iterations

X_resampled, y_resampled = renn.fit_resample(X, y)

#####################################################################
### set up the All-KNN undersampling class
#####################################################################

from imblearn.under_sampling import (
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN
)

allknn = AllKNN(
    sampling_strategy='auto',  # undersamples only the majority class
    n_neighbors=5, # the maximum size of the neighbourhood to examine
    kind_sel='all',  # all neighbours need to have the same label as the observation examined
    n_jobs=4)  # I have 4 cores in my laptop

X_resampled, y_resampled = allknn.fit_resample(X, y)


#####################################################################
### set up the Neighbourhood-Cleaning-Rule undersampling class
#####################################################################

from imblearn.under_sampling import NeighbourhoodCleaningRule
# set up Neighbourhood cleaning rule

ncr = NeighbourhoodCleaningRule(
    sampling_strategy='auto',# undersamples from all classes except minority
    n_neighbors=3, # explores 3 neighbours per observation
    kind_sel='all', # all neighbouring need to disagree, only applies to cleaning step
                    # alternatively, we can se this to mode, and then most neighbours
                    # need to disagree to be removed.
    n_jobs=4, # 4 processors in my laptop
    threshold_cleaning=0.5, # the threshold to evaluate a class for cleaning (used only for clearning step)
) 

# Note, the threshold_cleaning implementation by imbalanced-learn does not 
# match what was described in the original article. At the moment, it will
# only clean observations if the class has at least threshold * X.shape[0]
# number of observations

X_resampled, y_resampled = ncr.fit_resample(X, y)


#####################################################################
### set up the Neighbourhood-Cleaning-Rule undersampling class
#####################################################################

from imblearn.under_sampling import NearMiss

# set up Near Miss, first method
# that is, version = 1

nm1 = NearMiss(
    sampling_strategy='auto',  # undersamples only the majority class
    version=1,
    n_neighbors=3,
    n_jobs=4)  # I have 4 cores in my laptop

X_resampled, y_resampled = nm1.fit_resample(X, y)

# version = 2

nm2 = NearMiss(
    sampling_strategy='auto',  # undersamples only the majority class
    version=2,
    n_neighbors=3,
    n_jobs=4)  # I have 4 cores in my laptop

X_resampled, y_resampled = nm2.fit_resample(X, y)

# version = 3

nm3 = NearMiss(
    sampling_strategy='auto',  # undersamples only the majority class
    version=3,
    n_neighbors=3,
    n_jobs=4)  # I have 4 cores in my laptop

X_resampled, y_resampled = nm3.fit_resample(X, y)


#####################################################################
### set up the Instance Hardness Class undersampling class
#####################################################################

## instance hardness = 1 - p(1)

from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.multiclass import OneVsRestClassifier

# set up the random forests
# in a 1 vs rest manner

rf = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=10, random_state=1, max_depth=2),
    n_jobs=4,
)

probs = cross_val_predict(
    rf,
    X,
    y,
    cv=skf,
    n_jobs=4,
    method="predict_proba",
)

probs = pd.DataFrame(probs, columns=['p(0)', 'p(1)', 'p(2)'])

# to pick up a threshold, let's examine the probability
# distributions

probs.describe()

## We see that most observations of the majority class show p(0) > ~0.30. We can try and remove observations under this value. 
## This is arbitrary of course, we would have to try several thresholds and see what works best for our data.

# the 2 expressions are equivalent:

# remove obs. from majorty class where instance hardness ( 1 - p(0) ) is high
# remove obs. from majorty class where p(0) is low

# we remove observations from the majority (only) where the 
# random forests return probabilities below 0.35

condition_0 = (y==0) & (probs['p(0)'] <= 0.30)
condition_1 = (y==1) & (probs['p(1)'] <= 0.30)

# find those observations in the dataset
filtered_0 = X.loc[condition_0]
filtered_1 = X.loc[condition_1]

# number of observations to remove
filtered_0.shape, filtered_1.shape


# set up instance hardness threshold

iht = InstanceHardnessThreshold(
    estimator=rf,
    sampling_strategy='auto',  # undersamples all majority classes
    random_state=1,
    n_jobs=4, # have 4 processors in my laptop
    cv=3,  # cross validation fold
)

X_resampled, y_resampled = iht.fit_resample(X, y)



#####################################################################
### OVER SAMPLING
#####################################################################


#####################################################################
### Random Oversampling
#####################################################################

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(
    sampling_strategy='auto', # samples only the minority class
    random_state=0,  # for reproducibility
)  

X_res, y_res = ros.fit_resample(X, y)

# option 1: oversample all but majority class

ros = RandomOverSampler(
    sampling_strategy='not majority', # samples all but majority class
    random_state=0,  # for reproducibility
)  

X_res, y_res = ros.fit_resample(X, y)

# option 2: specify which classes to oversample

ros = RandomOverSampler(
    sampling_strategy={1:500, 2:500}, # we specify what to oversample
    random_state=0,  # for reproducibility
)  

X_res, y_res = ros.fit_resample(X, y)

#####################################################################
### Random Oversampling with smoothing
#####################################################################

from imblearn.over_sampling import RandomOverSampler

for shrink in [0, 0.5, 1, 10]:

    ros = RandomOverSampler(
        sampling_strategy='auto', # samples only the minority class
        random_state=0,  # for reproducibility
        shrinkage = shrink,
    )  

    X_res, y_res = ros.fit_resample(X, y)
    
    # plot of resampled data

    sns.scatterplot(
        data=X_res, x="VarA", y="VarB", hue=y_res, alpha=0.5
    )

    plt.title('Over-sampled dataset with shrinkage={}'.format(shrink))
    plt.show()
    
for shrink in [0, 0.5, 1, 10]:

    ros = RandomOverSampler(
        sampling_strategy='not majority', # samples all but majority class
        random_state=0,  # for reproducibility
        shrinkage = shrink,
    )  

    X_res, y_res = ros.fit_resample(X, y)

    # plot of resampled data

    sns.scatterplot(
        data=X_res, x="VarA", y="VarB", hue=y_res, alpha=0.5
    )

    plt.title('Over-sampled dataset with shrinkage={}'.format(shrink))
    plt.show()
    
 
#####################################################################
### SMOTE
#####################################################################

from imblearn.over_sampling import SMOTE

sm = SMOTE(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=5,
    n_jobs=4
)

X_res, y_res = sm.fit_resample(X, y)


#####################################################################
### SMOTE NC
#####################################################################

from imblearn.over_sampling import SMOTENC

smnc = SMOTENC(
    sampling_strategy='auto', # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=5,
    n_jobs=4,
    categorical_features=[2,3] # indeces of the columns of categorical variables
)  

X_res, y_res = smnc.fit_resample(X, y)



#####################################################################
### SMOTE N
#####################################################################

from imblearn.over_sampling import SMOTEN

# set up SMOTE

sampler = SMOTEN(
    sampling_strategy='auto', # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=5,
    n_jobs=4,
)

X_res, y_res = sampler.fit_resample(X, y)

#####################################################################
### ADASYN
#####################################################################

ada = ADASYN(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    n_neighbors=5,
    n_jobs=4
)

X_res, y_res = ada.fit_resample(X, y)

#####################################################################
### Borderline SMOTE
#####################################################################

from imblearn.over_sampling import BorderlineSMOTE


sm_b1 = BorderlineSMOTE(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=5, # the neighbours to crete the new examples
    m_neighbors=10, # the neiighbours to find the DANGER group
    kind='borderline-1',
    n_jobs=4
)

X_res_b1, y_res_b1 = sm_b1.fit_resample(X, y)

#####################################################################
### SVM SMOTE
#####################################################################


from imblearn.over_sampling import SVMSMOTE

sm = SVMSMOTE(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=5, # neighbours to create the synthetic examples
    m_neighbors=10, # neighbours to determine if minority class is in "danger"
    n_jobs=4,
    svm_estimator = svm.SVC(kernel='linear')
)

# remember that the templates are those minority observations
# within the danger zone

# create the synthetic examples
X_res, y_res = sm.fit_resample(X, y)


#####################################################################
### Kmean SMOTE
#####################################################################

from imblearn.over_sampling import KMeansSMOTE

sm = KMeansSMOTE(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=2,
    n_jobs=None,
    kmeans_estimator=KMeans(n_clusters=3, random_state=0),
    cluster_balance_threshold=0.1,
    density_exponent='auto'
)

X_res, y_res = sm.fit_resample(X, y)

### Comparing All

oversampler_dict = {

    'random': RandomOverSampler(
        sampling_strategy='auto',
        random_state=0),

    'smote': SMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        k_neighbors=5,
        n_jobs=4),

    'adasyn': ADASYN(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        n_neighbors=5,
        n_jobs=4),

    'border1': BorderlineSMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        k_neighbors=5,
        m_neighbors=10,
        kind='borderline-1',
        n_jobs=4),

    'border2': BorderlineSMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        k_neighbors=5,
        m_neighbors=10,
        kind='borderline-2',
        n_jobs=4),

    'svm': SVMSMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        k_neighbors=5,
        m_neighbors=10,
        n_jobs=4,
        svm_estimator=SVC(kernel='linear')),
}

def run_randomForests(X_train, X_test, y_train, y_test):

    rf = RandomForestClassifier(
        n_estimators=100, random_state=39, max_depth=2, n_jobs=4)
    rf.fit(X_train, y_train)

    print('Train set')
    pred = rf.predict_proba(X_train)
    print(
        'Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))

    print('Test set')
    pred = rf.predict_proba(X_test)
    print(
        'Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))

    return roc_auc_score(y_test, pred[:, 1])
  
  
# to save the results
results_dict = {}
shapes_dict = {}

for dataset in datasets_ls:
    
    results_dict[dataset] = {}
    shapes_dict[dataset] = {}
    
    print(dataset)
    
    # load dataset
    data = fetch_datasets()[dataset]
    
    # separate train and test
    X_train, X_test, y_train, y_test = train_test_split(
    data.data,  
    data.target, 
    test_size=0.3,
    random_state=0)
    
    # as some oversampling techniques use KNN
    # we set variables in the same scale
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
        
    roc = run_randomForests(X_train, X_test, y_train, y_test)
    
    # store results
    results_dict[dataset]['full_data'] = roc
    shapes_dict[dataset]['full_data'] = len(X_train)
    
    print()
    
    for oversampler in oversampler_dict.keys():
        
        print(oversampler)
        
        # resample
        X_resampled, y_resampled = oversampler_dict[oversampler].fit_resample(X_train, y_train)
        
        # evaluate performance
        roc = run_randomForests(X_resampled, X_test, y_resampled, y_test)
        
        #store results
        results_dict[dataset][oversampler] = roc
        shapes_dict[dataset][oversampler] = len(X_resampled)
        print()
        
    print()
        
