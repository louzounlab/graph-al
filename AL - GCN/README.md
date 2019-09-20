# Graph Active Learning

graph active learning using graph convolutional network (GCN)

## GCN
There are 4 models of GCN:
* Kipf - features are external data (content)
* Our Combined - features are combination of external data and topological attributes
* Our Topo Sym - features are topological attributes
* Our Topo aSymm - features are topological attributes + asymmetric matrix

For more details see Benami et al. "Topological based classification of paper domains using graph convolutional networks" (2019)

The model can be chosen at gcn.train.ModelRunner

To set the model's parameter look at gcn.train.build_model

## Active Learning
To run an active learning framework, you should create a explore_exploit.GraphExploreExploit object, and then use the run method.

A simple way to test several techniques of AL is to use main.active_learning. 
Parameters are: 
* data - The name of the data set. 
the data set must be in proj_dir\data_sets\name_of_the_data_set. 
this repository must contatin: 
tags.txt - file with labels. each line in a format of: node label 
graph_edges.txt - file with all edges. each line in a format of: n1,n2 (means an edge from node n1 to node n2 n1->n2). first line has to be "n1,n2"
content.txt (optional) - file with external information. each line in a format of: node x1 x2 x3 .... xn (where (x1,x2,...,xn) is the external data of the node)
* budget - if budget >= 1 the number of nodes, else the fraction from the whole data 
* batch - number of nodes to query each time 
* iterations - number of repetitions 
* out_interval - how many dots to display on the figures, 
* eps - list of desired epsilons. epsilon is the probability for exploring (choosing next nodes to query at random)

the desired methods must be specified inside the function using the main.run_active_model function.
The available methods are:
* entropy
* region_entropy
* rep_dist
* centrality
* Chang
* geo_dist
* geo_cent
* APR
* k_truss
* feature
* random
each method has sub-parameters which are described at explore_exploit. for example margin=True is used in region entropy to perform region margin.

For more details about the methods see Abel et al. "Regional based query in graph active learning" (2019)

## Code Example

```python
def active_learning(data, budget, batch=5, iterations=3, out_interval=25, eps=[0.05]):
    model = GraphExploreExploit(data, budget)
    params = [data, batch, iterations, out_interval]
    for epsilon in eps:
        run_active_model(model, 'random', *params, eps=epsilon)									# random
        run_active_model(model, 'entropy', *params, eps=epsilon)								# entropy
		run_active_model(model, 'region_entropy', *params, eps=epsilon)							# region entropy
		run_active_model(model, 'region_entropy', *params, eps=epsilon, **{'margin': True})		# region margin

    my_active_plot(Results, data)
    print(Results)

    return None


if __name__ == '__main__':
	Data_Sets = ['cora']
	for dataset in Data_Sets:
		active_learning(dataset, 0.15, out_interval=60, iterations=20, batch=1)
	
```
