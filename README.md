#Stochastic Weight Completion

This is a TensorFlow implementation of Stochastic Weight Completion for Road Networks, as described in our paper:
 
Jilin Hu, Chenjuan Guo, Bin Yang, Christian S. Jensen, [Stochastic Weight Completion for Road Networks Using Graph Convolutional Networks](https://ieeexplore.ieee.org/abstract/document/8731475) (ICDE 2019)

## Requirements
* tensorflow (>0.12)

## Run the demo

```bash
python gcrn_main_gcnn.py
```

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), 
* an N by D feature matrix (D is the number of features per node), and
* an N by E binary label matrix (E is the number of classes).

Have a look at the `BatchLoader` class in `utils.py` for an example.

You can specify a dataset as follows:

```bash
python gcrn_main_gcnn.py --server_name='chengdu'
```

(or by editing `gcrn_main_gcnn.py`)


## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{hu2019stochastic,
  title={Stochastic Weight Completion for Road Networks using Graph Convolutional Networks},
  author={Hu, Jilin and Guo, Chenjuan and Yang, Bin and Jensen, Christian S},
  booktitle={2019 IEEE 35th International Conference on Data Engineering (ICDE)},
  pages={1274--1285},
  year={2019},
  organization={IEEE}
}
```
