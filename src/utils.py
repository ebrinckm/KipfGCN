import numpy as np
import datetime
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import scipy.sparse
import cfscrape
from bs4 import BeautifulSoup

def test():
    print('Import succesful')

def merge(name):
    v = lambda x: (int(x)/1e18)
    g = lambda x: (int(x)/1e9)
    t = lambda x: datetime.datetime.fromtimestamp(int(x))

    # Concating all of my csv files together:
    files = ['C:/Data_Files/transactions/eth_txs-201901_0.csv','C:/Data_Files/transactions/eth_txs-201901_1.csv','C:/Data_Files/transactions/eth_txs-201901_2.csv']
    #      'C:/Data_Files/transactions/eth_txs-201901_3.csv','C:/Data_Files/transactions/eth_txs-201901_4.csv','C:/Data_Files/transactions/eth_txs-201901_5.csv',
    #      'C:/Data_Files/transactions/eth_txs-201901_6.csv','C:/Data_Files/transactions/eth_txs-201901_7.csv','C:/Data_Files/transactions/eth_txs-201901_8.csv']
    data = []
    i = 1
    for filename in files:
        dt = pd.read_csv(filename,
                    usecols=['hash','nonce','block_hash','block_number','transaction_index','from_address','to_address','value','gas','gas_price','block_timestamp'],
                    converters={'value': v, 
                                'gas_price': g, 
                                'gas': v,
                                'block_timestamp': t})
        data.append(dt.head(1000))
        print('File', i, 'loaded.')
        i += 1

    df = pd.concat(data, axis=0, ignore_index=True)
    return df
# The below merge is the original.
"""
def merge(name):
    This fucntion takes in the base filename of the dataset csv's.
    v = lambda x: (int(x)/1e18)
    g = lambda x: (int(x)/1e9)

    dfs = []

    # Change range back to 9 for all.
    for i in range(3):
        eths = pd.DataFrame(
            pd.read_csv(
                'C:/Data_Files/transactions/' + name + str(i) + '.csv',
                usecols=['hash','nonce','transaction_index','from_address','to_address','value','gas','gas_price'],
                converters={'value': v, 'gas_price': g, 'gas': v}
            )
        )
        # NOTE: Remove the .head() to enable full computation of the dataset.
        dfs.append(eths.head(1550))
        #dfs.append(eths.head(10000))
        #dfs.append(eths)
        print(int(((i+1)/9)*100), 'percent done merging.')
    return pd.concat(dfs)
"""
def create_feature_matrix(df):
    a=df.groupby(['from_address']).agg({
        'hash': ['count'],
        'value': ['min','max','mean','std']
    })
    a.columns = ['in_count', 'in_min', 'in_max', 'in_mean', 'in_std']
    a.index.name = 'address'

    b=df.groupby(['to_address']).agg({
        'hash': ['count'],
        'value': ['min','max','mean','std']
    })
    b.columns = ['out_count', 'out_min', 'out_max', 'out_mean', 'out_std']
    b.index.name = 'address'

    features = a.merge(b, how='outer', left_index=True, right_index=True)
    features.in_count = features.in_count.astype(float)
    features.out_count = features.out_count.astype(float)

    features = features.fillna(-1e-18)
    
    return features
# Below is the original creare feature matrix function.
"""
def create_feature_matrix(eth_df):
    This function takes in an Ethereum Dataframe and creates a feature matrix off of it.
    max_value = 11901464.23948 # the largest transaction ever, in ETH
    eth_df.value = eth_df.value/max_value
    max_gas = 508034850.0 # the highest gas total ever, in ETH
    eth_df.gas = eth_df.gas/max_gas
    max_price = 0.012487783 # the highest gas price ever, in ETH
    eth_df.gas_price = eth_df.gas_price/max_price

    a=eth_df.groupby(['from_address']).agg({
        'hash': ['count'],
        'value': ['min','max','mean','std']})
    a.columns = ['in_count', 'in_min', 'in_max', 'in_mean', 'in_std']
    a.index.name = 'address'


    b=eth_df.groupby(['to_address']).agg({
        'hash': ['count'],
        'value': ['min','max','mean','std']})
    b.columns = ['out_count', 'out_min', 'out_max', 'out_mean', 'out_std']
    b.index.name = 'address'

    features = a.merge(b, on='address')
    features.in_count = features.in_count.astype(float)
    features.out_count = features.out_count.astype(float)

    features = features.fillna(-1e-18)

    return features
"""
def getEtherScanInfo(address):
    scr =  cfscrape.create_scraper()
    page = scr.get(f'https://etherscan.io/address/{address}')
    soup = BeautifulSoup(page.text, 'html.parser')
    # Return account label
    label = soup.select_one('.h-100 .u-label')
    if label is not None:
        label = label.text.strip()
    # Return contract name
    stripped_contract = "EOA"
    contract_name = soup.select_one('div#code .mb-0')
    if contract_name is not None:
        stripped_contract = contract_name.text.strip()
        if stripped_contract == "0 ETH":
            stripped_contract = "EOA"
    return label, stripped_contract

def gen_labels(train, test):
    """This function takes in the training and testing dataframes to generates labels."""
    viable_train = []
    for i,a in enumerate(train.index):
        out = train.loc[a]['out_count']
        in_c = train.loc[a]['in_count']
        if(out + in_c) >= 200:
            viable_train.append((a,i))

    viable_test = []
    for i,a in enumerate(test.index):
        out = test.loc[a]['out_count']
        in_c = test.loc[a]['in_count']
        if(out + in_c) >= 200:
            viable_test.append((a,i))

    print('Generating training labels...')
    train_labels = []
    for addr in viable_train:
        label = getEtherScanInfo(addr[0])
        if(label is not None):
            train_labels.append(addr[1])
            
    print('Generatine testing labels...')
    test_labels = []
    for addr in viable_test:
        label = getEtherScanInfo(addr[0])
        if(label is not None):
            test_labels.append(addr[1])

    return train_labels, test_labels

def save_data(features, name):
    """This function takes in the features matrix and saves the data under the given name."""
    holder = features.copy()
    test_len = len(holder)
    y = np.zeros(shape=(len(holder),2), dtype=np.int32)
    for i in y:
        i[0] = 1
    
    X_train, X_test, y_train, y_test = train_test_split(holder, y, test_size=0.2)
    print('Searching for suspects...')
    train_l, test_l = gen_labels(X_train, X_test)

    for i in train_l:
        y_train[i][1] = 1
        y_train[i][0] = 0

    for j in test_l:
        y_test[j][1] = 1
        y_test[j][0] = 0
    print("Creating X...")
    #--Below is added for experimenting--
    X = pd.DataFrame(columns=(list(X_train.columns.values)))
    for i,r in enumerate(y_train):
        if(r[1] == 1):
            X.loc[len(X.index)] = (list(X_train.iloc[i].values))
    print(X.shape)
    #--//--
    print("...X Created")
    holder.drop(list(X_test.index), inplace=True)
    y = y_train
    
    # --X to scipy sparse--
    X = scipy.sparse.csc_matrix(X.values)
    X_train = scipy.sparse.csr_matrix(X_train.values)
    X_test = scipy.sparse.csr_matrix(X_test.values)
    #csr_features = scipy.sparse.csr_matrix(holder.values)

    i = (int(test_len * .8) + 1)
    index = []
    
    while(i <= test_len):
        index.append(int(i-1))
        i += 1

    print('Shape of features:', holder.shape) 
    print('Length of test index:', len(index))
    print('Shape of X_train and X_test data:', X_train.shape, X_test.shape)
    print('Shape of y_train and y_test data:', y_train.shape, y_test.shape)
    print('Shape of y:', y.shape)
    # -- Changed X_train -> X --
    print("Saving X_train...")
    pickle.dump(X, open('..\gcn\data\ind.' + name + '.x', 'wb'))
    print("... X_train saved")

    print("Saving X_test...")
    pickle.dump(X_test, open('..\gcn\data\ind.' + name + '.tx', 'wb'))
    print("... X_test saved")

    print("Saving all X...")
    pickle.dump(X_train, open('..\gcn\data\ind.' + name + '.allx', 'wb'))
    print("... all X saved.") 

    print("Saving y_train...")
    pickle.dump(y_train, open('..\gcn\data\ind.' + name + '.y', 'wb'))
    print("... y_train saved")

    print("Saving y_test...")
    pickle.dump(y_test, open('..\gcn\data\ind.' + name + '.ty', 'wb'))
    print("... y_test saved")

    print("Saving all y...")
    pickle.dump(y_train, open('..\gcn\data\ind.' + name + '.ally', 'wb'))
    print("... all y saved.")

    print("Saving test index...")
    pickle.dump(index, open('..\gcn\data\ind.' + name + '.test.index', 'wb'))
    print("... test index saved.")
    
