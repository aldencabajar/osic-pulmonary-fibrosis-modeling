import tensorflow_probability as tfp
import seaborn as sns
import pandas as pd
import pydicom
import os
import re
import time
import gzip
from tqdm import tqdm
import seaborn as sns
import sklearn.preprocessing
import tensorflow as tf 
tfd = tfp.distributions
tfb = tfp.bijectors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import ct_scan_utils.ct_scan_processing2 as ct
import osic_utils
from tqdm import tqdm_notebook as tqdm



def make_joint_dist_coroutine(num_ids, pids, features):

    """
    Creates a two-level hierarchical model for variational inference
    """
    def model():
        X_cst = tf.cast(features, tf.float32)
        # random intercepts
        Root = tfd.JointDistributionCoroutine.Root
        patient_scale = yield Root(tfd.HalfCauchy(loc = 0, scale = 5.))
        intercept = yield Root(tfd.Normal(loc = 0, scale = 10.))
        patient_prior = yield tfd.MultivariateNormalDiag(loc = tf.zeros(num_ids), 
                                                       scale_identity_multiplier = patient_scale)
        int_resp = tf.gather(patient_prior, pids, axis = -1) + intercept[...,tf.newaxis]
        
        # random slopes for week var
        beta_week = yield Root(tfd.Normal(loc = 0, scale = 10.))
        beta_week_scale = yield Root(tfd.HalfCauchy(loc = 0, scale = 5.))
        beta_week_prior = yield tfd.MultivariateNormalDiag(loc = tf.zeros(num_ids),
                                                          scale_identity_multiplier = beta_week_scale)
        bw_resp = (tf.gather(beta_week_prior, pids, axis = -1)  + beta_week[..., tf.newaxis]) * X_cst[:,0]
        
        # other variables
        betas = yield Root(tfd.MultivariateNormalDiag(loc = tf.zeros(tf.shape(X_cst)[1] - 1), 
                                                scale_identity_multiplier = 10.))
        other_vars_resp = tf.tensordot(betas, tf.transpose(X_cst[:,1:]), axes = 1)    
        total_response = int_resp + bw_resp + other_vars_resp
        
        # response scale
        resp_scale = yield Root(tfd.HalfCauchy(loc =0, scale = 5.))
        
        yield tfd.Normal(loc = total_response,  scale = resp_scale[...,tf.newaxis])
        
    return tfd.JointDistributionCoroutineAutoBatched(model)

# declare template variables for loc and scale parameters
_init_loc = lambda shape=(): tf.Variable(
    tf.random.uniform(shape, minval=-2., maxval=2.))

_init_scale = lambda shape=(): tfp.util.TransformedVariable(
    initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
    bijector=tfb.Softplus())

def make_surrogate_posterior(num_ids, num_vars):
    return(
         tfd.JointDistributionSequentialAutoBatched([
                  tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),                       # scale_prior
                  tfd.Normal(_init_loc(), _init_scale()),                                       # intercept
                  tfd.Normal(_init_loc(shape = [num_ids]), _init_scale(shape = [num_ids])),      # patient prior
                  tfd.Normal(_init_loc(), _init_scale()),                                       # week random slope
                  tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),                       # week scale prior
                  tfd.Normal(_init_loc(shape = [num_ids]), _init_scale(shape = [num_ids])),      # patient by week prior
                  tfd.Normal(_init_loc(shape = [num_vars]), 
                             _init_scale(shape = [num_vars])),                            # other vars prior
                  tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),                       # response scale
                    
                ])
    )


if __name__ == '__main__':
    # loading data
    train_data = pd.read_pickle('data/train_preproc_struct_data.pkl')
    test_data = pd.read_pickle('data/test_preproc_struct_data.pkl')
    lung_stats = pd.read_csv('data/lung_statistics.csv')

    # divide by 10^3 to reduce the scale of response
    lung_stats['lung_vol'] = lung_stats['lung_vol']/int(10e3)

    # get ids, feature matrix, and  labels from data
    remove_cols = [
        'Patient', 'Weeks', 'FVC', 'Percent',
        'baseline', 'pct_bsl', 'Percent', 'id_no'
        ]
    choose_cols = [
        'Weeks_std', 'Age', 'Ex-smoker',
        'Never smoked', 'Sex_', 'baseline_std', 'pct_bsl_std'
        ]

    X = train_data.drop(columns = remove_cols).loc[:, choose_cols]
    ids = train_data['id_no'].values
    labels =  train_data['FVC'].values

    X_test = test_data(columns = remove_cols).loc[:, choose_cols]
    ids_test = test_data['id_no'].values
    labels_test =  train_data['FVC'].values

    # Create joint distribution
    joint_distr = make_joint_dist_coroutine(
        num_ids = 176, 
        pids = tf.constant(ids, tf.int32),
        features = X 
        )
   ##### RUN OPTIMIZATION OF PARAMETERS  #######
    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    num_ids = len(ids)

    def target_log_prob_fn(*args):
        return joint_distr.log_prob(*args, labels/1000)

    surrogate_posterior = make_surrogate_posterior(num_ids, X.shape[1] - 1) 

    start = time.time()

    losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn,  
    surrogate_posterior,
    optimizer=optimizer,
    num_steps= 100000, 
    seed=42,
    sample_size= 500
    )

    end = time.time()

    print("processing time:", str((end - start)/60), "min")

    # sample from posteriors
    (scale_prior_, 
    intercept_,  
    patient_weights,
    week_slope,
    week_scale_prior,
    week_patient_weights,
    other_vars_weights,
    response_scale), _ = surrogate_posterior.sample_distributions()



    














