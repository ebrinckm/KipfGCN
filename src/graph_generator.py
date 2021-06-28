import collections
import sys
import pickle
import utils

def gen_graph(features, eth_df, name):
    hold = features.to_dict()
    addrs = list(list(hold.values())[0].keys())

    first_occ = []

    count = 0
    for i in addrs:
        first_occ.append((count, i))
        count += 1

    def get_index(address):
        return(addrs.index(address))

    graph = collections.defaultdict(list)

    for j in first_occ:
        for k in range(len(eth_df['from_address'])):
            if(j[1] == eth_df['from_address'].values[k]):
                addr = eth_df['to_address'].values[k]
                if(addr in addrs):
                    index = get_index(addr)
                    graph[j[0]].append(index)
                else:
                    # Fill in the blank nodes with itself.
                    graph[j[0]].append(j[0])
    
    print(len(graph)) 
    print("Saving graph...")
    pickle.dump(graph, open('..\gcn\data\ind.' + name + '.graph', 'wb'))
    print("... graph saved.")
    
    return graph

def list_to_graph(dataset_str):
    """This function takes in the name of the graph file as made be Andrey and converts
       to meet Kipf and Welling's requirements."""
    with open("ind.{}.{}".format(dataset_str, 'graph'), 'rb') as f:
        if sys.version_info > (3, 0):
            aslist = pickle.load(f, encoding='latin1')
        else:
            aslist = pickle.load(f)

    graph = collections.defaultdict(list)

    for i in aslist:
        graph[i[0]] = i[1]

    return graph

