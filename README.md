# RCFusion
Original implementation of the paper "Recurrent Convolutional Fusion for RGB-D Object Recognition": https://arxiv.org/pdf/1806.01673.pdf


（主要修改了code/train_and_eval.py文件中的checkpoint保存和读取）
数据集2400/1400张。Epoch0时loss为4，准确率0.03，1时3，0.54。2时2，0.8。5时1.35，0.84。50时0.39，0.9。（训练一个epoch大概半小时） 

eval.py显示验证集的分类结果。

txt.py将文件夹中所有图片名和标签1生成一个txt文件。

crop.py将文件夹中所有图片裁剪。


### Requirements:
* Tensorflow 1.10.0
* CUDA 8.0 （可以用cpu运行）
* Python 2.7
* See python requirements in requirements.txt
* Docker + nvidia-docker [optional]: https://github.com/NVIDIA/nvidia-docker

### Instructions:
1. Download dataset and parameteres (see link below) and extract them in directory <dataset_dir> and <params_dir>
[Skip to point (4) to run w/o docker] 下载数据集和模型，若不使用docker跳到第4点。
2. To execute the code within a docker container, run ```docker build -t <container_name> .```
3. Start the container with ```docker run -it --runtime=nvidia -v <dataset_dir>:<dataset_dir> -v <params_dir>:<params_dir> <container_name> bash```
4. Run ```python code/train_and_eval.py <dataset_dir> <params_dir>```
（自己电脑上使用cpu：```python code/train_and_eval.py /Users/Vita/PycharmProjects/rcfusion-master/ocid_dataset /Users/Vita/PycharmProjects/rcfusion-master/resnet18_ocid_params```

5.验证准确率：```python code/eval.py /Users/Vita/PycharmProjects/rcfusion-master/ocid_dataset /Users/Vita/PycharmProjects/rcfusion-master/resnet18_ocid_params```


6.显示图片和预测类别：```python code/test.py /Users/Vita/PycharmProjects/rcfusion-master/ocid_dataset /Users/Vita/PycharmProjects/rcfusion-master/resnet18_ocid_params```
（在test_rgb1和test_hha1上做验证！验证不同图片需要替换文件夹中图片！每次需要更改test_labels1.txt中的信息，需要改最后显示的图片路径！）


### Disclaimers:
* The paper should be cosidered the main reference for this work. All the details of the algorithm and the training are reported there.
* The data augmentation taken from an external repo. Credits go to: https://github.com/aleju/imgaug
* WARNING: code has been developed w/ Tensorflow 1.5.0. We noticed some fluctuation in the results when migrating to Tensorflow 1.10.0. 

### Download:
* Pre-processed semantic crops of OCID dataset: https://data.acin.tuwien.ac.at/index.php/s/e46X2cCIjLXoRn7
* Original semantic crops of OCID dataset (we recommend using the link above instead to have the exact same pre-training we used in our experiments): https://www.acin.tuwien.ac.at/vision-for-robotics/software-tools/object-clutter-indoor-dataset/
* ResNet-18 pre-trained weights for initialization: https://data.acin.tuwien.ac.at/index.php/s/RueHQUbs2JtoHeJ

### Contributors:
* Mohammad Reza Loghmani - email: loghmani@acin.tuwien.ac.at
* Mirco Planamente - email: mirco.planamente@iit.it

### Citation:
```
@ARTICLE{rcfusion, 
author={M. R. {Loghmani} and M. {Planamente} and B. {Caputo} and M. {Vincze}}, 
journal={IEEE Robotics and Automation Letters}, 
title={Recurrent Convolutional Fusion for RGB-D Object Recognition}, 
year={2019}, 
volume={4}, 
number={3}, 
pages={2878-2885}, 
doi={10.1109/LRA.2019.2921506}}
```
