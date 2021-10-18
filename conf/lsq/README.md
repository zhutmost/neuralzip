# LSQ-Net: Learned Step Size Quantization

## Introduction

LSQ-Net is proposed by Steven K. Esser and et al. from IBM. It can be found on [arXiv:1902.08153](https://arxiv.org/abs/1902.08153).

Here are some experiment results.

| Network   | Quan. Method | Bitwidth (W/A)  | Top-1 Acc. (%) | Top-5 Acc. (%) |
|:---------:|:------------:|:---------------:|:--------------:|:--------------:|
| ResNet-18 |          LSQ |             2/2 |          65.37 |          86.37 |
| ResNet-18 |          LSQ |             3/3 |          68.75 |          88.91 |
| ResNet-18 |          LSQ |             4/4 |          69.97 |          89.32 |
| ResNet-50 |          LSQ |             2/2 |          68.40 |          88.27 |
| ResNet-50 |          LSQ |             3/3 |          75.42 |          92.62 |
| ResNet-50 |          LSQ |             4/4 |          76.23 |          92.94 |

## Run Command

Run the following command to quantize pre-trained ResNet with vairous configurations.

We only provide [one configuration](https://github.com/zhutmost/neuralzip/blob/master/conf/lsq/resnet18-imagenet-a3w3.yaml) for 3-bit ResNet-18 quantization, and you can override its paramters in CLI to perform quantization with other parameters.

Except for the batch size, my implementation is the same as the original paper. I use a batch size of 128 per device for ResNet-18, and the paper uses 64.

### ResNet-18 (act: 2b, wgt: 2b)

```bash
python main.py conf_filepath=./conf/lsq/resnet18-imagenet-a3w3.yaml quan.act.params.bit=2 quan.weight.params.bit=2 experiment_name='resnet18-imagenet-a2w2-lsq'
```
### ResNet-18 (act: 3b, wgt: 3b)

```bash
python main.py conf_filepath=./conf/lsq/resnet18-imagenet-a3w3.yaml
```
### ResNet-18 (act: 4b, wgt: 4b)

```bash
python main.py conf_filepath=./conf/lsq/resnet18-imagenet-a3w3.yaml quan.act.params.bit=4 quan.weight.params.bit=4 experiment_name='resnet18-imagenet-a4w4-lsq'
```
### ResNet-50 (act: 2b, wgt: 2b)

```bash
python main.py conf_filepath=./conf/lsq/resnet18-imagenet-a3w3.yaml quan.act.params.bit=2 quan.weight.params.bit=2 experiment_name='resnet50-imagenet-a2w2-lsq' model.class_name=resnet50 dataset.batch_size=64
```
### ResNet-50 (act: 3b, wgt: 3b)

```bash
python main.py conf_filepath=./conf/lsq/resnet18-imagenet-a3w3.yaml experiment_name='resnet50-imagenet-a3w3-lsq' model.class_name=resnet50 dataset.batch_size=64
```
### ResNet-50 (act: 4b, wgt: 4b)

```bash
python main.py conf_filepath=./conf/lsq/resnet18-imagenet-a3w3.yaml quan.act.params.bit=4 quan.weight.params.bit=4 experiment_name='resnet50-imagenet-a4w4-lsq' model.class_name=resnet50 dataset.batch_size=64
```
