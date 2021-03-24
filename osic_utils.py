import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def preprocess_structured_data(data, pid_dict, encoder_dicts, return_df = True):
    """
        A function to preprocess the structured data.
        data = a pandas dataframe with the structured data
        pid dict = dictionary of patients with id num
        scaler_dicts = a dictionary of encoders for standardizing and encoding
    """
    required_enc = np.array(['SmokingStatus', 'Age', 'Weeks', 'pct_bsl', 'baseline'])
    bools = np.array([i in encoder_dicts.keys() for i in required_enc])
    missing_enc = np.where(bools == False)[0]

    assert missing_enc.shape[0] == 0, \
    "Please specify " + ', '.join(list(required_enc[missing_enc])) + ' encoders'

    ### subset the data into the required pids, aggregate duplicate data
    data = data.groupby(
        list(data.columns[~data.columns.isin(["FVC", "Percent"])])
    ).agg({'FVC' : 'mean',
       'Percent' :'mean'}).reset_index()
    data['id_no'] = data.merge(
        pd.Series(pid_dict).to_frame() \
        .reset_index() \
        .rename(columns = {'index': 'Patient', 0: 'id_no'}),
        on = "Patient"
    )['id_no']

    ### One hot encoding categorical variables ####
    # smoking status
    oh_encoder = encoder_dicts['SmokingStatus']
    smoking_one_hot_encode = list(oh_encoder.categories_[0])
    data[smoking_one_hot_encode] = pd.DataFrame(oh_encoder.transform(
        data['SmokingStatus'].values.reshape(-1, 1)))

    # remove a level from smoking status to avoid non-full rank matrix
    data.drop(columns = ['Currently smokes'], inplace = True)

    # gender
    data['Sex_'] = np.where(data['Sex'] == 'Male', 1, 0)

    # drop other columns
    data.drop(columns = ['Sex', 'SmokingStatus'], inplace = True)

    # Standardize Age
    age_scaler = encoder_dicts['Age']
    data['Age'] = pd.DataFrame(
        age_scaler.transform(
    data['Age'].values.reshape(-1,1)))

    data.sort_values(['Patient', 'Weeks'], inplace = True)

    # Normalize weeks by their actual start weeks
    data['Weeks'] = data['Weeks'] - data.groupby(['Patient'])['Weeks'].transform(min)

    # esatablish baseline fvc and percentages for each patient
    data['baseline'] = pd.merge(
        data[['Patient']],
        data.loc[data['Weeks'] == 0][['Patient', 'FVC']],
        on = "Patient", how = 'left')['FVC']

    data['pct_bsl'] = pd.merge(
        data[['Patient']],
        data.loc[data['Weeks'] == 0][['Patient', 'Percent']],
        on = "Patient", how = "left")['Percent']

    # Standardize weeks, baseline fvc, and percentage baseline
    for col in ['Weeks', 'baseline', 'pct_bsl']:
        colname = col + '_std'
        data[colname] = encoder_dicts[col] \
        .transform(
            data[col].values.reshape(-1,1))

    #return data.drop(columns = ['Patient', 'Weeks'])

    X = data.drop(columns = ['Patient', 'Weeks', 'FVC', 'Percent',
                             'baseline', 'pct_bsl', 'Percent', 'id_no']
                 )[["Weeks_std", "Age", "Ex-smoker",
                    "Never smoked", "Sex_", "baseline_std", "pct_bsl_std"]]
    print('columns in X:', ', '.join(X.columns))
    #X_tmp = data['Weeks_std'].values
    ids = data['id_no'].values
    y =  data['FVC'].values

    if return_df:
        return data

    return X.values, ids, y


class DatasetGen:
    def __init__(self, pid_dict, split, seed, root_dir, batch_size = 32):
        """pid = list of Patient IDs
           split = Train %
           seed = random seed
        """
        np.random.seed(seed)
        n = len(pid_dict)
        keys = list(pid_dict.keys())
        trid = np.random.choice(keys, int(n * split), replace = False)
        tsid = np.setdiff1d(keys, trid)


        self.train = tf.data.Dataset.from_generator(
            lambda: [(root_dir + k + "/" ,pid_dict[k]) for k in trid],
        (tf.string, tf.int32)).shuffle(200).batch(batch_size)
        self.test = tf.data.Dataset.from_generator(
            lambda: [(k, pid_dict[k]) for k in tsid],
        (tf.string, tf.int32))

    def add_struct_data(self, struct_data, id_tensor, y):
        """Adds structured data"""

        def add(id_str, id_, struct_data, id_tensor, y):
            idx = [tf.squeeze(tf.where(id_tensor == i)) for i in id_]
            feat = tf.concat([tf.gather(struct_data, i, axis = 0) for i in idx], axis = 0)
            label = tf.concat([tf.gather(y, i, axis = 0) for i in idx], axis = 0)
            idx = tf.concat([tf.ones_like(j) * i for i, j in enumerate(idx)], axis = 0)

            return id_str, id_, feat, label, idx

        ds = self.train.map(lambda id_str, x:
                            tf.py_function(add, [id_str, x, struct_data, id_tensor, y],
                                           Tout = [tf.string, tf.int32, tf.float64, tf.float64, tf.int64]))
        ds_test = self.test.map(lambda id_str, x:
                                tf.py_function(add, [id_str, x, struct_data, id_tensor, y],
                                                Tout = [tf.string, tf.int32, tf.float64, tf.float64, tf.int64]))

        self.train = ds
        self.test = ds_test

    def add_img_data(self):

        def add(id_str, idx):
            imgs = [ct.image_preprocess(i) for i in id_str]

            # calculating lung volume
            lung_vol_out = []
            for masked_img, st, ps, lps in imgs:
                lung_vol_out.append(
                ct.get_volume(masked_img, tf.constant(st), tf.constant(ps), lps)
                )

            # calculating lung image statistics
            stats = []
            for img, _ in lung_vol_out:
                stats.append(
                    ct.calculate_statistics(tf.constant(img))[None, ...])
            stats = tf.concat(stats, axis = 0)

            ### return lung volume, mean, variance, skew and kurtosis in that order
            # divide by the magnitude to decrease range
            lung_volume = tf.concat([i[1][None, ...] for i in lung_vol_out], axis = 0) /1e5
            all_feats = tf.cast(
                tf.concat([lung_volume, stats], axis = 1),
                dtype = tf.float32)
            all_feats = tf.gather(all_feats, idx, axis = 0)

            return all_feats

        ds = self.train.map(lambda id_str, _, __, lbl, idx:
                               (_, __, lbl, idx, tf.py_function(add, [id_str, idx], tf.float32))
                           )
        self.train = ds




if __name__ == "__main__":
    train_data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
    test_data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

    patient_dcm_dict = {}
    root_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"
    for dirname, _, filenames in os.walk(root_dir):
        if 'ID' in dirname:
            dirname_ = dirname.replace(root_dir, "")
            patient_dcm_dict[dirname_] = filenames
    patient_dcm_dict = {k: i for i, k in enumerate(sorted(patient_dcm_dict.keys()))}


    ### fit encoders before running preprocessing data ###
    # one-hot encoder
    oh_encoder = OneHotEncoder(sparse = False, dtype = 'int32')
    oh_encoder.fit(train_data['SmokingStatus'].values.reshape(-1, 1))

    ### Standard scalers ####
    # AGE
    age_scaler = StandardScaler()
    age_scaler.fit(train_data[['Patient', 'Age']].drop_duplicates()['Age'] \
                   .values.reshape(-1, 1))
    # WEEKS
    week_scaler = StandardScaler()
    week_scaler.fit(train_data[['Patient', 'Weeks']].drop_duplicates()['Weeks'] \
                   .values.reshape(-1,1))
    # PERCENTAGE
    pct_scaler = StandardScaler()
    pct_scaler.fit(train_data['Percent'].values.reshape(-1,1))
    # BASELINE FVC
    baseline_scaler = StandardScaler()
    baseline_scaler.fit(train_data['FVC'].values.reshape(-1,1))

    encoder_dict = {
        'Age': age_scaler,
        'Weeks': week_scaler,
        'pct_bsl': pct_scaler,
        'baseline': baseline_scaler,
        'SmokingStatus': oh_encoder
    }
    train_data_processed = preprocess_structured_data(
        train_data, patient_dcm_dict,
        encoder_dict, return_df = True
    )

    test_data_processed = preprocess_structured_data(
        test_data, patient_dcm_dict,
        encoder_dict, return_df = True
    )


    if 'data' not in os.listdir():
        os.mkdir('data')
    train_data_processed.to_pickle('data/train_preproc_struct_data.pkl')
    test_data_processed.to_pickle('data/test_preproc_struct_data.pkl')

    
