# Leveraging Multi-facet Paths for Heterogeneous Graph Representation Learning (MF2Vec)



### Overview

### Requirements
- Python version: 3.9
- scikit-learn
- dgl = 1.1.2
- numpy = 1.26.2
- pandas = 2.1.0
- torch = 2.0.1

### How to Run
````

conda create -n py39 python=3.9
source activate py39
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scikit-learn & Requirements
````

# Execute MF2vec on Filmtrust dataset
````
python main.py --dataset dblp 
````

### Arguments

````--dataset````: name of the dataset

````--isInit````: If ````True````, warm-up step is performed

````--dim````: dimension size of a node

````--lr````: learning rate

````--patience````: when to stop (early stop criterion)

````--isReg````: enable aspect regularization framework

````--reg_coef````: lambda in aspect regularization framework

````--num_aspects````: number of predefined aspects (K)
