import numpy as np
import pandas as pd
import pystan
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

def transform_test_data(test_data):
    test_data_ = pd.concat([test_data[['Patient']]]*146).sort_values('Patient')
    test_data_['Weeks_tr'] = test_data_.groupby('Patient').cumcount() - 12
    test_data_ = pd.merge(test_data_, test_data, how = "left", on = "Patient")
    test_data_.rename(columns = {'Weeks': 'Weeks_base'}, inplace = True)
    test_data_['Weeks']= test_data_['Weeks_tr'] - test_data_['Weeks_base']
    test_data_.drop(columns = ['Weeks_tr'], inplace = True)

    return test_data_


def convert_to_tensor(data, train = True):
    patient_no = tf.convert_to_tensor(data['patient_no'], dtype = tf.int32)
    fvc = tf.convert_to_tensor([data'FVC'].values, dtype = tf.float32)   
    X = tf.convert_to_tensor(data[['Sex_', 'Ex-smoker', 'Never smoked', 
                                 'Age', 'baseline_std', 'pct_bsl_std']].values, 
                                 dtype = tf.float32)
    X_weeks = tf.convert_to_tensor(data['Weeks'].values, dtype = tf.float32)
     
    if train:
        return patient_no, fvc, X, X_weeks, df
    else:
        return fvc, X, X_weeks, df

def affine(X, kernel_diag):
    kernel_diag = tf.ones_like(X) * kernel_diag
    return X * kernel_diag

def predict_from_posterior(X, X_weeks, fit):
    num_draws = len(fit['sigma_y'])
    pred = affine(X_weeks[...,None], fit['bw'][None, ...]) \
    + affine(tf.ones_like(X_weeks[...,None]), fit['a0'][None, ...]) \
    + affine(X[:,4][...,None], fit['b_bsline'][None, ...]) \
    + affine(X[:,3][..., None], fit['b_age'][None, ...])  \
    + affine(X[:,0][...,None], fit['b_gender'][None, ...]) \
    + affine(X[:,1][...,None], fit['b_ex_smoker'][None, ...]) \
    + affine(X[:,2][...,None], fit['b_never_smoked'][None, ...]) \
    + affine(X[:,5][...,None], fit['b_pct'][None, ...]) \
    + X[:,4][...,None] * X_weeks[...,None] * fit['b_bsline_w'][None, ...] \
    + X[:,5][...,None] * X_weeks[...,None] * fit['b_pct_w'][None, ...] \
    
    # noise_std = fit['sigma_y'][None, ...] * np.random.randn(X_weeks.shape[0], num_draws)
    # pred_with_noise = pred + noise_std
    # fvc_mean_pred = tf.reduce_mean(pred_with_noise, axis = 1)
    # fvc_mean_sd = tf.math.reduce_std(pred_with_noise, axis= 1)
    
    return pred.numpy(), fit['sigma_y'][None, ...].numpy()




if __name__ == '__main__':
    train_data_processed = pd.read_pickle('data/train_preproc_struct_data.pkl')
    test_data_processed = pd.read_pickle('data/test_preproc_struct_data.pkl')
    test_transformed =  transform_test_data(test_data_processed)

    # prepare training data
    patient_no, fvc_train, X_train, X_weeks_train = convert_to_tensor(train_data_processed) 
    # prepare test data
    fvc_test, X_test, X_weeks_test = convert_to_tensor(test_transformed) 


    ##### MODEL CODE #############################
    model_code = """
        data {
            int <lower=0> N;
            int <lower=0> L;
            int <lower=0, upper=L> pid[N];
            int <lower=0> p;
            vector[N] X_w;
            vector[N] y;
            matrix[N, p] X;
        }

        parameters {
            real a0;
            real bw; //
            vector[L] a;
            vector[L] b;
            real b_bsline;
            real b_bsline_w; //coef for bsline and week interaction
            real b_pct;
            real b_pct_w;
            real  b_age;
            real b_gender;
            real b_ex_smoker;
            real b_never_smoked;
            real <lower=0> sigma0;
            real <lower=0> sigma_bw;
            real <lower=0> sigma_y;
        }

        transformed parameters {

            vector[N] yhat;
            vector[N] m;
            vector[N] g;
            for (i in 1:N) {
                m[i] = b[pid[i]] + b_bsline_w * X[i,5] + b_pct_w * X[i, 6]; // for random slopes with week
                g[i] = a[pid[i]] +  b_bsline * X[i,5] + b_age * X[i,4] + b_gender * X[i,1] + 
                b_ex_smoker * X[i,2] + b_never_smoked * X[i,3] + b_pct * X[i,6];
                yhat[i] = g[i] + m[i] * X_w[i];
            }
        }

        model {
            a0 ~ normal(0, 1e5);
            bw ~ normal(0, 1e5);
            b_bsline ~ normal(0, 1e5);
            b_bsline_w ~ normal(0, 1e5);
            b_age ~ normal(0, 1e5);
            b_gender ~ normal(0, 1e5);
            b_ex_smoker ~ normal(0, 1e5);
            b_never_smoked ~ normal(0, 1e5);
            b_pct ~ normal(0, 1e5);
            b_pct_w ~ normal(0, 1e5);
            a ~ normal(a0, sigma0);
            b ~ normal(bw, sigma_bw);
            y ~ normal(yhat, sigma_y);
            
        }


        """
    data = {'N': train_data_processed.shape[0], 
            'L': np.unique(patient_no.numpy()).shape[0],
            'pid': patient_no.numpy() + 1,
            'X_w': X_weeks_train.numpy(),
            'p': tf.shape(X_train).numpy()[1],
            'y': fvc_train.numpy(),
            'X': X_train.numpy()}

    # start sampling from model using data
    mdl = pystan.StanModel( model_code = model_code, verbose = True)
    m_sample = mdl.sampling(data = data, iter=num_iter, chains = 4)
    print(m_sample)


    # we now predict from the posterior distribution
    fvc_mean_train, sd_train = predict_from_posterior(X, X_weeks, m_sample)
    fvc_mean_test, sd_test = predict_from_posterior(X_test, X_weeks_test, m_sample)
    train_data_processed['fvc_pred'] = fvc_mean_train
    train_data_processed['Confidence'] = sd_train

    test_transformed['FVC'] = fvc_mean_test
    test_transformed['Confidence'] = sd_test

    test_transformed['Patient_Week'] = test_transformed[['Patient', 'Weeks', 'Weeks_base']] \
                                       .apply(
                                            lambda x: '{}_{}'.format(x['Patient'], x['Weeks'] + x['Weeks_base']), 
                                            axis = 1
                                       )

    submission = test_transformed[['Patient_Week',  'FVC', 'Confidence']]
    submission.to_csv("data/submission.csv", index = False)









