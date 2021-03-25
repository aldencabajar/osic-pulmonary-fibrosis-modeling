import matplotlib
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import seaborn as sns
import pydicom
import os
import re
import time
import gzip
from tqdm import tqdm
import seaborn as sns
import sklearn.preprocessing
tfd = tfp.distributions
tfb = tfp.bijectors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import ct_scan_processing2_module as ct
import osic_utils
from tqdm import tqdm_notebook as tqdm



def make_joint_dist_coroutine(num_ids, pids, X):
    def model():
        X_cst = tf.cast(X, tf.float32)
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