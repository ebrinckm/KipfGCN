import utils
import graph_generator
import pickle
"""The purpose of this program is to piece together other programs to 
   create and then save the needed data using the large ethereum dataset."""
dataset_str = 'tobedeleted'
# Load in our full dataset.
eth_df = utils.merge('eth_txs-201901_')
# Extract our features.
print('Creating features...')
features = utils.create_feature_matrix(eth_df)
print('Preprocessing and saving the data...')
utils.save_data(features, dataset_str)
print('Creating graph...')
graph = graph_generator.gen_graph(features, eth_df, dataset_str)
