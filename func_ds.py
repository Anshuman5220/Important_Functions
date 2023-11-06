
## Label encoding (One hot encoding)
df = pd.get_dummies(df, columns=['col_name'], prefix=['prefix_cols'])

## Binary encoding
df['flag'] = df['flag'].map({'Yes': 1, 'No': 0})

## Custom describe with missing values in it

def custom_describe(df):
    summary = df.describe(percentiles=[0.01,0.1,0.2,0.5,0.9,0.99]).T
    missing_values = df.isna().sum().T
    missing_stats = pd.DataFrame({
        'missing_values': missing_values,
        'percent_missing': (missing_values / len(df)) * 100
    })
    return pd.concat([summary, missing_stats], axis=1)

## Plotting bar chart for categorical variables

def PlotBarCharts(inpData, colsToPlot):
    %matplotlib inline
    import matplotlib.pyplot as plt
    # Generating multiple subplots
    fig, subPlot=plt.subplots(nrows=1, ncols=len(colsToPlot), figsize=(20,5))
    fig.suptitle('Bar charts of: '+ str(colsToPlot))

    for colName, plotNumber in zip(colsToPlot, range(len(colsToPlot))):
        inpData.groupby(colName).size().plot(kind='bar',ax=subPlot[plotNumber])

PlotBarCharts(inpData=df, colsToPlot=['col1', 'col2', 'col3'])

## Plotting histogram:
df.hist(['var1', 'var2', 'var3'], figsize=(18,10))

## Missing Value Treatment
def missing_fit(df,num,cat,version):
    dtypes_num = pd.DataFrame({"col":num,"dtype":"num"})
    dtypes_cat = pd.DataFrame({"col":cat,"dtype":"cat"})
    dtypes = pd.concat([dtypes_num,dtypes_cat],ignore_index=True)
    dtypes['default_impute_type'] = np.where(dtypes.dtype=='num','median',np.where(dtypes.dtype=='cat','mode',np.nan))
    dtypes = _custom_impute(df,dtypes,'default')
    dtypes.to_csv('saved_results/impute_{}.csv'.format(version),index=False)
    impute_vals = dtypes.default_impute_value
    impute_vals.index = dtypes.col
    return impute_vals

def missing_transform(df,impute_vals):
    if impute_vals is None:
        raise("Not fitted error")
    df = df.fillna(impute_vals)
    return df

def _custom_impute(df,dtypes,impute_id):
    dtype2 = dtypes[dtypes.dtype.isin(['num','cat'])].reset_index(drop=True)
    if dtype2[impute_id+'_impute_type'].isnull().sum()>0:
        print(impute_id+'_impute_type contains missing values')
    impute_dict = {dtype2['col'][i]:dtype2[impute_id+'_impute_type'][i] for i in range(dtype2.shape[0])}
    impute_values = {}
    for key,val in impute_dict.items():
        impute_values[key]=aggregate(df,key,val)
    dtypes[impute_id + '_impute_value'] = dtypes.col.map(impute_values)
    return dtypes

def aggregate(df,key,val):
    if val=='mean':
        return df[key].mean()
    elif val=='median':
        return df[key].median()
    elif val == 'mode':
        return df[key].mode()[0]

## Outlier Treatment
def outlier_fit(df,num,outlier_cap=0.99,version=None):
    dtypes_num=num
    out_treat_df = pd.DataFrame({'Variable':dtypes_num})
    out_treat_df['outlier_cap']=outlier_cap
    out_treat_df.to_csv('saved_results/out_treat_df.csv',index=False)
    outlier_treatment_df = pd.DataFrame(columns=['var_name','outlier_cap','Max','Upper Bound','Lower Bound','Min'])
    for var in dtypes_num:
        _out_cap =  out_treat_df[out_treat_df.Variable==var]["outlier_cap"].values[0]
        temp_df = pd.DataFrame(data={'var_name':var,'outlier_cap':_out_cap,'Max':df[var].max(),'Upper Bound':df[var].quantile(_out_cap),'Lower Bound':df[var].quantile(1-_out_cap),'Min':df[var].min()},columns=['var_name','outlier_cap','Max','Upper Bound','Lower Bound','Min'],index=[var])
        outlier_treatment_df = outlier_treatment_df.append(temp_df)
    outlier_treatment_df.to_csv('saved_results/outlier_treatment_df.csv',index=False)
    
def outlier_transform(df):
    outlier_treatment_df = pd.read_csv('saved_results/outlier_treatment_df.csv',index_col='var_name')
    for var in outlier_treatment_df.index:
        _upper_bound=outlier_treatment_df[outlier_treatment_df==var]["Upper Bound"].values[0]
        _lower_bound=outlier_treatment_df[outlier_treatment_df==var]["Lower Bound"].values[0]
        df.loc[df[var] > _upper_bound,var] = _upper_bound
        df.loc[df[var] < _lower_bound,var] = _lower_bound
    return df


## Tfidf features in case of seq variable 

def fit_tfidf(seq,sam_frac,dset):
    print('Fitting tfidf ...')
    tfidf_vectorizer = TfidfVectorizer(stop_words=None,ngram_range=(1,3),analyzer='char',lowercase=False,min_df=0.005)
    tfidf_matrix = tfidf_vectorizer.fit_transform(seq)
    joblib.dump(tfidf_vectorizer,'saved_objects/tfidf_vectorizer.joblib',compress=1)
    return tfidf_matrix

def fit_feature_selection(df,y,undersample_frac,dset,perc=10):
    print("Fitting feature selection")
    selector = SelectPercentile(f_classif,percentile=perc)
    output = selector.transform(df,y)
    joblib.dump(selector,'saved_objects/selector.joblib',compress=1)
    output_dense = pd.DataFrame(output.todense())
    tfidf_vectorizer = joblib.load('saved_objects/tfidf_vectorizer.joblib')
    support = np.asarray(selector.get_support())
    columns = np.asanyarray(list(tfidf_vectorizer.get_feature_names()))
    output_dense.columns = columns[support]
    return(output_dense)

## Correlation:

def print_highly_correlated(df, threshold=0.95):
	"""Prints highly correlated features pairs in the data frame (helpful for feature engineering)"""
	corr_df = df.corr() # get correlations
	correlated_features = np.where(np.abs(corr_df) > threshold) # select ones above the abs threshold
	correlated_features = [(corr_df.iloc[x,y], x, y) for x, y in zip(*correlated_features) if x != y and x < y] # avoid duplication
	s_corr_list = sorted(correlated_features, key=lambda x: -abs(x[0])) # sort by correlation value
	
	if s_corr_list == []:
		print("There are no highly correlated features with correlation above", threshold)
	else:
		for v, i, j in s_corr_list:
			cols = df.columns
			print ("%s and %s = %.3f" % (corr_df.index[i], corr_df.columns[j], v))
            
print_highly_correlated(model_data)

## Check 0 variance features:
## Checking for 0 variance feature
from sklearn.feature_selection import VarianceThreshold
feature_selector = VarianceThreshold(threshold=0)
feature_selector.fit(model_data)
[x for x in model_data.columns if x not in model_data.columns[feature_selector.get_support()]]
    
#Remove quasi-constant features
sel = VarianceThreshold(threshold=0.01)  
# fit finds the features with low variance
sel.fit(model_data)  
# how many not quasi-constant?
sum(sel.get_support())
print(model_data.shape)
features_to_keep = list(model_data.columns[sel.get_support()])
model_data = model_data[features_to_keep]
print(model_data.shape)


## Re compute missing values/outlier analysis for selected features from original dataset

impute_vals=None
version='v1'
cat=[]
id=[]
target=['actual_total_time']
rm_cols = cat+id+target
cols = list(train_1.columns)
num = []

for x in cols:
    if x not in rm_cols:
        num.append(x)
        
impute_vals = missing_fit(train_1,num,cat,version)
imputed_train1 = missing_transform(train_1,impute_vals)
imputed_test1 = missing_transform(test_1,impute_vals)


outlier_fit(imputed_train,num,version=version)
model_data = outlier_transform(imputed_train1)
model_data_test = outlier_transform(imputed_test1)

## Standardization and normalisation 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform="pandas")
# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)
# transform train and test sets
X_train_scaled = scaler.transform(X_train)

from sklearn.preprocessing import MinMaxScaler

# set up the scaler
scaler = MinMaxScaler().set_output(transform="pandas")
# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


## ML Model
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from scipy.special import logit
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#os.chdir('')
#os.getcwd()
#df = pd.read_csv()
#
#x_train,x_test,y_train,y_test = train_test_split(df.drop('target',axis=1),df['target'],test_size=0.3,random_state=42)


from xgboost import XGBRegressor

def default_xgboost_model(model_num,xdev,ydev,xitv,yitv):
    if model_num !=0:
        feature_list = pd.read_csv('feature_reduction_iteration'+str(model_num-1)+'.csv',header=0,sep=',')
        feature_list.Variable = feature_list.Variable.apply(lambda x:x.strip())
        xdev=xdev[feature_list[feature_list.CumGain_Prop<=0.95].Variable.values]
        xitv=xitv[feature_list[feature_list.CumGain_Prop<=0.95].Variable.values]
    model =  XGBRegressor(objective='reg:squarederror',silent=True,metrics=['rmse'],njobs=6,random_state=0)
    model.fit(xdev,ydev)
    ## Saving the model
    joblib.dump(model,'default_model_'+str(model_num)+'.joblib')
    ## Make predictions
    pred_dev = model.predict(xdev)
    pred_itv = model.predict(xitv)
    ## Rsq for both itv and otv
    rsq_dev = metrics.r2_score(ydev,pred_dev,sample_weight=None)
    rsq_itv = metrics.r2_score(yitv,pred_itv,sample_weight=None)
    rmse_dev = metrics.mean_squared_error(ydev,pred_dev,sample_weight=None)
    rmse_itv = metrics.mean_squared_error(ydev,pred_dev,sample_weight=None)
    ## Obtain Feature Importance
    num_features = len(xdev.columns)
    feature_dict = model.get_booster().get_score(importance_type='gain')
    feature_df = pd.DataFrame.from_dict(feature_dict,orient='index',columns=['Gain']).sort_values(['Gain'],ascending=False)
    feature_df['CumGain'] = feature_df.Gain.cumsum()
    feature_df['CumGain_Prop'] = feature_df.CumGain/max(feature_df.CumGain)
    feature_df['Variable'] = feature_df.index
    feature_df.to_csv('feature_reduction_iteration'+str(model_num)+'.csv',header=True,index=False)
    global result_df
    result_df = result_df.append(pd.DataFrame({'Model_No':model_num,'rsq_dev':rsq_dev,'rsq_itv':rsq_itv,"Ttl_features":num_features},index=[0]))


result_df = pd.DataFrame()
iter=10

for i in range(itr):
    print("Running Iteration {0}".format(i))
    default_xgboost_model(i,X_train,y_train,X_test,y_test)
result_df.to_csv('Default_Model_Iteration.csv',index=False)


from itertools import product,chain

def expand_grid(dict_df):
    return pd.DataFrame([row for row in product(*dict_df.values())],columns=dict_df.keys())

param_grid = {
    'max_depth':[2,3,4],
    'gamma':[1],
    'n_estimators':[200,400,500],
    'learning_rate':[0.01,0.05],
    'min_child_weight':[1,3],
    'colsample_bytree':[0.8,1],
    'sub_sample':[0.6,0.8,1.0]
}

param_df = expand_grid(param_grid)
param_df


## Optuna

import optuna
from sklearn.model_selection import cross_val_score, train_test_split

def objective(trial):

    rf_n_estimators = trial.suggest_int("rf_n_estimators", 100, 1000)
    rf_criterion = trial.suggest_categorical("rf_criterion", ['squared_error'])
    rf_max_depth = trial.suggest_int("rf_max_depth", 1, 4)
    rf_min_samples_split = trial.suggest_float("rf_min_samples_split", 0.01, 1)
    
    model = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        criterion=rf_criterion,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
    )

    score = cross_val_score(model, X_train, y_train, cv=3)
    accuracy = score.mean()
    return accuracy

study = optuna.create_study(
    direction = 'maximize',
    sampler=optuna.samplers.RandomSampler(),
)

study.optimize(objective, n_trials=5)
study.best_params

optuna.visualization.plot_param_importances(study)

## Get All in one dataframe

trial_numbers = []
hyperparameters_list = []
objective_values = []

# Access information for all trials
for trial in study.trials:
    trial_number = trial.number
    hyperparameters = trial.params
    objective_value = trial.value

    trial_numbers.append(trial_number)
    hyperparameters_list.append(hyperparameters)
    objective_values.append(objective_value)

# Create a Pandas DataFrame from the lists
data = {
    'Trial Number': trial_numbers,
    'Hyperparameters': hyperparameters_list,
    'Objective Value': objective_values
}
all_iterations = pd.DataFrame(data)

# Print the DataFrame
print(all_iterations)


## KS table

def ks_table(score,response,indentifier):
    print("getting KS...")
    group=10
    df = pd.DataFrame({'score':score,'response':response})
    df = df.sort_values(['score'],ascending=[False])
    bin_size= len(score) % group
    if rem == 0:
        df['groups'] = list(np.repeat(range(rem+1,11),bin_size))
    else: 
        df['groups'] = list(np.repeat(range(1,rem + 1),bin_size+1)) + list(np.repeat(range(rem+1,11),bin_size))
    grouped = df.groupby('groups',as_index=False)
    agg = pd.DataFrame({'Total_Obs':grouped.count().response})
    agg['No.Res'] = grouped.sum().response
    agg['No.Non_Res']=agg['Total_Obs'] - agg['No.Res']
    agg['min_pred']= grouped.min().score
    agg['max_pred'] = grouped.max().score
    agg['pred_rr'] = grouped.mean().score
    agg['cum_no_res'] = agg['No.Res'].cumsum()
    agg['cum_no_non_res'] = agg['No.Non_Res'].cumsum()
    agg['percent_cum_res'] = agg['cum_no_res'] / agg['cum_no_res'].max()
    agg['percent_cum_non_res'] = agg['cum_no_non_res'] / agg['cum_no_non_res'].max()
    agg['KS'] = agg['percent_cum_res'] - agg['cum_no_non_res']
    agg.to_csv('saved_results/KS_table_'+indentifier+'.csv',index=False)
    return(agg)
    
## Find Optimal cutoff

def find_optimal_cutoff(target,predicted):
    ## Optimal cutoff will be where tpr is high and fpr is low 
    ## tpr - (1-fpr) is 0 or near to zero would be ideal cutoff point
    fpr, tpr, threshold = roc_curve(target,predicted)
    i = np.arrange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1-fpr),index=i),'threshold':pd.Series(threshold,index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])
    

### Feature Iteration Dev
from xgboost import XGBClassifier

def saveFeatureIterationDev(df, id, target, version=None,save_pred=False):
    id_var=id
    y = target
    version = version
    print("First Trail run with all features ..")
    out = df.loc[:,id_var]
    out['actual'] = df[y[0]]
    model = XGBClassifier(seed=10)
    drop_vars = id_var + y
    model.fit(df.drop(drop_vars,axis=1),df[y[0]])
    imp_features_df = pd.DataFrame({'feature_names':df.drop(drop_vars,axis=1).columns,'importance':model.feature_importance_})
    imp_features_df.sort_values('importance',ascending=False,inplace=True)
    imp_features_df.to_csv('saved_results/feature_importance_'+version+'.csv',index=False)
    if save_pred:
        pred = model.predict_proba(df.drop(drop_vars,axis=1))
        out['pred'] = pred
        out.to_csv('saved_results/pred_dev_overall_'+version+'.csv',index=False)
    print("Iterating on different feature combination on Dev")
    summary_df = pd.DataFrame()
    imp_features_df = pd.read_csv('saved_results/feature_importance_'+version+'.csv')
    imp_features_df = imp_features_df.sort_values('importance',ascending=False)
    imp_features = imp_features_df[imp_features_df['importance'] !=0]
    feature_count = len(imp_features)
    if feature_count > 100:
        iter = list(range(10,100,10)) + list(range(100,feature_count,50)) + [feature_count]
    else:
        iter = list(range(10,feature_count,10)) + [feature_count]
        
    target = df[y[0]]
    for i in iter:
        print('Feature count {0}'.format(i))
        curr_features = imp_features.feature_names[:i]
        curr_X = df.loc[:,curr_features]
        model= XGBClassifier(seed=10)
        model.fit(curr_X,target)
        joblib.dump(model,'saved_objects/xgb_'str(i) + '_features_'+version + '.joblib',compress=1)
        if save_pred:
            pred = model.predict_proba(df.drop(drop_vars,axis=1))
            out['pred'] = pred[:1]
            out.to_csv('saved_results/pred_dev_'+str(i) + '_features_'+version+'.csv',index=False)
        feature_imp = pd.DataFrame({'feature_names':curr_features,'importance':model.feature_importance_})
        feature_imp.to_csv('saved_results/feature_importance_'+str(i)+'_features_'+version+'.csv',index=False)
        score = model.predict_proba(curr_X)
        ks = ksTable(score[:,1],target,'dev_xgb_'+str(i) + '_features_'+version)
        breaks= np.diff(ks['No.Res'])>0
        dec_breaks = (np.diff(ks['No.Res'])>0).any()
        ks_val = ks.KS.max()
        ks_decile = ks.KS.idxmax() + 1
        capture = ks['percent_cum_res'][3]
        if dec_break:
            break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
            summary_df = summary_df.append(pd.DataFrame([[i,ks_val,break_dec,ks_decile,capture]],columns=['feature_count','dev_ks','dev_ro_break','dev_ks_decile','dev_capture']))
        else:
            break_dec = np.nan
            summary_df = summary_df.append(pd.DataFrame([[i,ks_val,break_dec,ks_decile,capture]],columns=['feature_count','dev_ks','dev_ro_break','dev_ks_decile','dev_capture']))
            
    summary_df.to_csv('saved_results/summary_df_features_xgb_'+version+'.csv',index=False)
    

def saveFeatureIterationVal(df,id,target,dset,version=None,save_pred=False):
    print('Iterating on different feature combination in {} ..'.format(dset))
    id_var=id
    y = target
    version = version
    out = df.loc[:,id_var]
    out['actual'] = df[y[0]]
    summary_df_test = pd.DataFrame()
    print("In test feature iter function ...")
    imp_feature_df = pd.read_csv('saved_results/feature_importance_'+version+'.csv')
    imp_features_df = imp_features_df.sort_values('importance',ascending=False)
    imp_features = imp_features_df[imp_features_df['importance'] !=0]
    feature_count = len(imp_features)
    if feature_count > 100:
        iter = list(range(10,100,10)) + list(range(100,feature_count,50)) + [feature_count]
    else:
        iter = list(range(10,feature_count,10)) + [feature_count]
        
    target = df[y[0]]
    summary_df = pd.read_csv('saved_results/summary_df_features_xgb_' + version + '.csv')
    for i in iter:
        print(dset + 'iteration {}'.format(i))
        curr_features = imp_features.feature_names[:i]
        curr_X = df.loc[:,curr_features]
        model = joblib.load('saved_objects/xgb_' + str(i) + '_features_' + version + '.joblib')
        if save_pred:
            pred = model.predict_proba(curr_X)
            out['pred'] = pred[:,1]
            out.to_csv('saved_results/pred_'+dset+'_'+str(i) + '_features_' + version + '.csv',index=False)
        score = model.predict_proba(curr_X)
        ks = ksTable(score[:,1],target,dset+'_xgb_'+str(i) + '_features_' + version + '.csv',index=False)
        breaks= np.diff(ks['No.Res'])>0
        dec_breaks = (np.diff(ks['No.Res'])>0).any()
        ks_val = ks.KS.max()
        ks_decile = ks.KS.idxmax() + 1
        capture = ks['percent_cum_res'][3]
        if dec_break:
            break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
            summary_df_test = summary_df_test.append(pd.DataFrame([[i,ks_val,break_dec,ks_decile,capture]],columns=['feature_count',dset + '_ks',dset + '_ro_break',dset +'_ks_decile',dset + '_capture']))
        else:
            break_dec = np.nan
            summary_df_test = summary_df_test.append(pd.DataFrame([[i,ks_val,break_dec,ks_decile,capture]],columns=['feature_count',dset + '_ks',dset + '_ro_break',dset + '_ks_decile',dset + '_capture']))
            
    summary_df_test.reset_index(drop=True,inplace=True)
    summary_df[dset+'_ks'] = summary_df_test[dset + '_ks']
    summary_df[dset+'_ro_break'] =  summary_df_test[dset + '_ro_break']
    summary_df[dset+'_ks_decile'] =  summary_df_test[dset + '_ks_decile']
    summary_df[dset+'_capture'] =  summary_df_test[dset + '_capture']
    summary_df['dev_'+dset+'_ks_diff'] = (summary_df['dev_ks'] - summary_df[dset + '_ks'])*100/summary_df['dev_ks']
    summary_df.to_csv('saved_results/summary_df_features_xgb_'+version + '.csv',index=False)
    

def featureIterationSummary(version=None):
    iter_type = 'features'
    identifier = 'xgb_' + version
    summary_df = pd.read_csv('saved_results/summary_df_'+iter_type+'_'+identifier + '.csv')
    summary_df['itv_otv_ks_diff'] = (summary_df['itv_ks'] - summary_df['otv_ks'])*100/summary_df['itv_ks']
    summary_df['dev_otv_diff_cat'] = (summary_df['dev_otv_ks_diff'] <= 10,1,0)
    summary_df['otv_ro_cat'] = np.where(summary_df['otv_ro_break'].fillna(11)>7,1,0)
    summary_df['dev_ro_cat'] = np.where(summary_df['dev_ro_break'].fillna(11)>7,1,0)
    summary_df['itv_ro_cat'] = np.where(summary_df['itv_ro_break'].fillna(11)>7,1,0)
    cols = ['dev_otv_diff_cat','otv_ro_cat','itv_ro_cat','dev_ro_cat']
    tups = summary_df[cols].sort_values(cols,ascending=False).apply(tuple,1)
    f,i = pd.factorize(tups)
    factorized = pd.Series(f+1,tups.index)
    summary_df = summary_df.assign(Rank1=factorized)
    tups2 = summary_df.loc[:,['Rank1','otv_ks']].sort_values(['Rank1','otv_ks'].ascending=[True,False]).apply(tuple,1)
    f2, i2 = pd.factorize(tups2)
    factorized2 = pd.Series(f2+1,tups2.index)
    summary_df = summary_df.assign(Rank2 = factorized2)
    
    summary_df['dev_itv_ks_diff_score'] = 100 - abs(summary_df['dev_itv_ks_diff'])
    summary_df['dev_otv_ks_diff_score'] = 100 - abs(summary_df['dev_otv_ks_diff'])
    summary_df['itv_otv_ks_diff_score'] = 100 - abs(summary_df['itv_otv_ks_diff'])
    
    summary_df['dev_ro_score'] = 100*summary_df['dev_ro_break'].fillna(11)/11
    summary_df['itv_ro_score'] = 100*summary_df['itv_ro_break'].fillna(11)/11
    summary_df['otv_ro_score'] = 100*summary_df['otv_ro_break'].fillna(11)/11
    
    summary_df['stability_score'] = (summary_df['dev_itv_ks_diff_score']+summary_df['dev_otv_ks_diff_score'] + summary_df['itv_otv_ks_diff_score'] + summary_df['dev_ro_score'] + summary_df['itv_ro_score'] + summary_df['otv_ro_score'])/6
    summary_df['stability_weighted_otv_ks'] = summary_df['stability_score']*summary_df['otv_ks']
    summary_df.sort_values['stability_weighted_otv_ks',ascending=False,inplace=True)
    summary_df.to_csv('saved_results/summary_df_'+iter_type + '_' + identifier + '_ordered.csv',index=False)
    return summary_df
    
dictionary = {}

def put_ParamSpace():
    default_param_grid = {
                        'n_estimators':[100,200]
    
    
def saveTrainingDev(df,id,target,imp_features,param_df,version=None,save_pred=False):
    print('Iterating on different hyper parameters ..')
    version = version
    out = df.loc[:,id]
    out['actual'] = df[target[0]]
    summary_df = pd.DataFrame()
    identifier = str(len(imp_features)) + 'var'
    alias = {
            'n_estimators':'est',
            'max_depth':'max_dep',
            'subsample':'sub_s',
            'learning_rate':'learn_r',
            'colsample_bytree':'col_samp',
            'reg_lambda':'lambda'
            }
            
    for idx, row in params_df.astype(object).iterrows():
        print('Iteration {0} of {1}'.format(idx+1,params_df.shape[0]))
        tup = [i for i in zip([alias.get(row.index[j]) for j in range(len(params_df.columns))],row.values.astype(str))]
        params_str = [''.join(t) for t in tup]
        identifier = indentifier + '_'.join(params_str) + '_' + version
        param = row.to_dict()
        model = XGBClassifier(seed=10,learning_rate=param['learning_rate'],colsample_bytree=param['colsample_bytree'],n_estimators=param['n_estimators'],subsample=param['subsample'],max_depth=param['max_depth'],nthread=10)
        model.fit(df.loc[:,imp_features],df[target[0]])
        joblib.dump(model,'saved_objects/xgb_'+identifier)
        feature_imp = pd.DataFrame({'feature_names':imp_features,'importance':model.feature_importance_})
        feature_imp.to_csv('saved_results/pred_dev_'+identifier+'.csv',index=False)
        score = model.predict_proba(df.loc[:,imp_features])
        if save_pred:
            out['pred']= score[:,1]
            out.to_csv('saved_results/pred_dev_'+indentifier+'.csv',index=False)
        ks = ksTable(score[:,1],df[target[0]],'dev_xgb_'+identifier)
        breaks= np.diff(ks['No.Res'])>0
        dec_breaks = (np.diff(ks['No.Res'])>0).any()
        ks_val = ks.KS.max()
        ks_decile = ks.KS.idxmax() + 1
        capture = ks['percent_cum_res'][3]
        if dec_break:
            break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
            summary_df = summary_df.append(pd.DataFrame(list(row.values) + [ks_val,break_dec,ks_decile,capture],columns=list(row.index) + ['dev_ks','dev_ro_break','dev_ks_decile','dev_capture'))
        else:
            break_dec = np.nan
            summary_df = summary_df.append(pd.DataFrame(list(row.values) + [ks_val,break_dec,ks_decile,capture],columns=list(row.index) + ['dev_ks','dev_ro_break','dev_ks_decile','dev_capture']))
        identifier = str(len(imp_features)) + 'var'
    sumamry_df.to_csv('saved_results/summary_df_params_xgb_' + version + '.csv',index = False)
    
    
def saveTestingVal(df,id,target,imp_features,dset,params_df,version=None,save_pred = False):
    print('Applying parameter iteration on {0}'.format(dset))
    version= version
    params_df = params_df
    out = df.loc[:,id]
    out['actual'] = df[target[0]]
    summary_df_test = pd.DataFrame()
    summary_df = pd.read_csv('saved_results/summary_df_params_xgb_'+version+'.csv')
    identifier = str(len(imp_features)) + 'var'
    alias = {
            'n_estimators':'est',
            'max_depth':'max_dep',
            'subsample':'sub_s',
            'learning_rate':'learn_r',
            'colsample_bytree':'col_samp',
            'reg_lambda' : 'lambda'
            }
    for idx, row in params_df.astype(object).iterrows():
        print('Iteration {0} of {1}'.format(idx+1,params_df.shape[0]))
        tup = [i for i in zip([alias.get(row.index[j]) for j in range(len(params_df.columns))],row.values.astype(str))]
        params_str = [''.join(t) for t in tup]
        identifier = indentifier + '_'.join(params_df) + '_' + version
        param = row.to_dict()
        model = joblib.load('saved_objects/xgb_'+indentifier)
        score = model.predict_proba(df.loc[:,imp_features])
        if save_pred:
            out['pred'] = score[:,1]
            out.to_csv('saved_results/pred_' + dset + '_' + indentifier + '.csv',index=False)
        ks = ksTable(score[:,1],df[target[0]],dset + '_xgb_' + identifer)
        breaks= np.diff(ks['No.Res'])>0
        dec_breaks = (np.diff(ks['No.Res'])>0).any()
        ks_val = ks.KS.max()
        ks_decile = ks.KS.idxmax() + 1
        capture = ks['percent_cum_res'][3]
        if dec_break:
            break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
            summary_df_test = summary_df_test.append(pd.DataFrame(list(row.values) + [ks_val,break_dec,ks_decile,capture],columns=list(row.index) + [dset + '_ks',dset + '_ro_break',dset + '_ks_decile',dset + '_capture'))
        else:
            break_dec = np.nan
            summary_df_test = summary_df_test.append(pd.DataFrame(list(row.values) + [ks_val,break_dec,ks_decile,capture],columns=list(row.index) + [dset + '_ks',dset + '_ro_break',dset + '_ks_decile',dset + '_capture'))
        indentifier = str(len(imp_features)) + 'var'
    summary_df_test.reset_index(drop=True,inplace=True)
    summary_df[dset + '_ks'] = summary_df_test[dset + '_ks']
    summary_df[dset + '_ro_break'] = summary_df_test[dset + '_ro_break']
    summary_df[dset + '_ks_decile'] = summary_df_test[dset + '_ks_decile']
    summary_df[dset + '_capture'] = summary_df_test[dset + '_capture']
    summary_df['dev_' + dset + '_ks_diff'] = (summary_df['dev_ks'] - summary_df[dset + '_ks'])*100/summary_df['dev_ks']
    sumamry_df.to_csv('saved_results/summary_df_params_xgb_'+version + '.csv',index=False)
    

version = 'v1'
impute_vals = None
cat = ['seq']
id = ['clnt_nbr','crd_acct_nbr']
target = ['response']
rm_cols = cat + id + target
cols = list(train.columns)
num = []
for x in cols:
    if x not in rm_cols:
        num.append(x)

impute_vals = fit(train,num,cat,version=version)
imputed_train = transform(train,impute_vals)


outlier.fit(imputed_data)
model_data = outlier_transform(imputed_train)

saveFeatureIterationDev(model_data,id,target,version='v1',save_pred = False)
saveFeatureIterationVal(itv_data,id,target,'itv',version='v1')

dictionary= {}


## Formatting Excel:

from pandas import ExcelWriter

def format_excel(df,sheet_nm,writer,startrow,startcol,text):
    df.to_excel(writer,sheet_name='{0}',format(sheet_nm),startrow=startrow,startcol=startcol,index=False)
    workbook= writer.book
    worksheet = writer.sheet['{0}'.format(sheet_nm)]
    header_format = workbook.add_format({'bold':True,'text_wrap':True,'valign':'top','fg_color':'#AED6F1','border':1})
    border_format = workbook.add_format({'border':1,'border_color':'#000000'})
    worksheet.conditional_format(startrow,startcol,startrow+df.shape[0],startcol + df.shape[1]-1,{'type':'no_blanks','format':border_format)
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(startrow,col_num+startrow,value,header_format)
    worksheet.write(startrow,startcol,text)
    
writer = ExcelWriter()


## Pyspark Window Functions:

from pyspark.sql import Window

wspec1 = Window.partitionBy('ACCT_NBR','bill_cycle')

df_core_anr = df_core_anr.WithColumn('rev_core_anr',sum(col('rev_core_anr')).over(wspec1)).WithColumn('max_period',max(col('period')).over(wspec2))


    
