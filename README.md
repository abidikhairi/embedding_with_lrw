# Lazy Random Walks Network Embedding

We applied a tweaked verion of lazy random walk to generate walks from a given graph, then fed these walks to skipgram to learn nodes representations.

## Lazy Random Walks

### Original Version
![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Clarge%20w_%7Bij%7D%20%3D%20exp%28-%20%5Cfrac%7B%7C%7Cg_%7Bi%7D%20-%20g_%7Bj%7D%7C%7C%5E2%7D%7B2%5Csigma%5E2%7D%29)

![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Clarge%20d_%7Bi%7D%20%3D%20%5Csum_%7Bj%20%5Cin%20V%7D%7Bw_%7Bij%7D%7D)

![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Clarge%20%5Cmathcal%7BP%7D_%7Bij%7D%20%3D%20%5Cleft%5C%7B%20%5Cbegin%7Barray%7D%7Bll%7D%201%20-%20%5Calpha%20%26%20%5Cmbox%7Bif%20%7D%20i%20%3D%20j%20%5C%5C%20%5Calpha%20w_%7Bij%7D%20/%20d_%7Bi%7D%20%26%20%5Cmbox%7Bif%20%7D%20%28i%2C%20j%29%20%5Cin%20E%20%5C%5C%200%20%26%20otherwise%20%5Cend%7Barray%7D%20%5Cright.)

### Tweaked Version
We adapt the previously defined weight between any given nodes as follows:

- `g` is the feature vector instead of pixel intensity (original paper).
- we extend the `L2` norm to be any user defined similarity function.

### Project Requirements

- scikit-learn
- dgl
- wandb
- matplotlib
- numpy
- node2vec
- gensim

## References
```
@ARTICLE{6725608,
  author={Shen, Jianbing and Du, Yunfan and Wang, Wenguan and Li, Xuelong},
  journal={IEEE Transactions on Image Processing}, 
  title={Lazy Random Walks for Superpixel Segmentation}, 
  year={2014},
  volume={23},
  number={4},
  pages={1451-1462},
  doi={10.1109/TIP.2014.2302892}
}
```
