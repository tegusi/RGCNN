## Regularized Graph CNN for Point Cloud Segmentation

This code is the tensorflow implementation of our preprinted paper, [RGCNN: Regularized Graph CNN for Point Cloud Segmentation][arxiv], ACM MultiMedia, 2018. Camera-ready version will be updated soon.

## Installation

This code runs on tensorflow 1.4 and python 3.6 with additional library such as h5py. We borrow the framework of [cnn_graph][cng].

## Usage

It requires original ModelNet40 and ShapeNet data, which can be downloaded [here][data_seg] for segmentation and [here][data_cls] for classification. You can use the tool provided by [pointnet][pointnet++] to convert the data to numpy array. We also provide our [processed one][data_pre] but we don't guarantee its compatibility.

The train.py is quite easy to read, I'm sure you can run and test it smoothly.

## Note

We test our code on jupyter notebook at first, so I suspect some part could be missing. If something went wrong, please contact me by email.

[cng]: https://github.com/mdeff/cnn_graph
[arxiv]: https://arxiv.org/abs/1806.02952
[data_seg]: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
[data_cls]: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
[pointnet++]: https://github.com/charlesq34/pointnet2
[data_pre]: https://1drv.ms/f/s!Am_uh1epJzCIjQeZviRjHa4fCkFy