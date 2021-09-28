<h1 align=center> NeuralZip </h1>
<div align="center">

**Compress Your Neural Network Painlessly**

</div>

## Introduction

NeuralZip is a out-of-box Python scaffold for neural network quantization-aware training (QAT).
With NeuralZip, you can focus on optimizing your quantization algorithm itself without falling into error-prone and dirty jobs.

NeuralZip can:
- Inject quantizer operators into your network without modification
- Decouple your quantizer implementation with the network implementation
- Built-in quantizers, including LSQ/...
- Evaluate your quantization algorithm on common benchmarks, including ImageNet/CIFAR10/...
- Automatic accelerate training with multi-thread DistributedDataParallel
- Checkpoint / TensorBoard visualization / YAML & CLI configuration / Thorough logs

## Example Cases

Here are some LSQ-based experiment results. Their configuration files can be found in the [conf](./conf) folder.

| Network   | Quan. Method | Bitwidth (W/A)  | Top-1 Acc. (%) | Top-5 Acc. (%) |
|:---------:|:------------:|:---------------:|:--------------:|:--------------:|
| ResNet-18 |          LSQ |             2/2 |          65.37 |          86.37 |
| ResNet-18 |          LSQ |             3/3 |          68.96 |          88.56 |
| ResNet-18 |          LSQ |             4/4 |          69.97 |          89.32 |

The only implementation difference from the origin paper is that my batch size is 128 per GPU card.

## User Guide

### Install Dependencies

Install library dependencies within an Anaconda environment.

```bash
# Create a environment with Python 3.9
conda create -n neuralzip python=3.9
conda activate neuralzip
# PyTorch GPU version >= 1.9
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
# PyTorch Lightning & its Bolts
conda install pytorch-lightning -c conda-forge
conda install torchmetrics
pip install lightning-bolts
# Miscellaneous
conda install omegaconf gym -c conda-forge
conda install scikit-learn
conda update --all
```

### Run Scripts with Your Configurations

This program use YAML files as inputs. A template as well as the default configuration is provided as `conf/template.yaml`.
Please read it before running Python scripts. It is disallowed to modify this file and use it as your configuration, which may cause severe errors at the runtime.

If you want to change the behaviour of this program, please copy it somewhere else. And then run the `main.py` with your modified configuration file.

```bash
python main.py conf_filepath=/path/to/your/config/file.yaml
```

The modified options in your YAML file will overwrite the default settings. For details, please read the comments in `conf/template.yaml`.
You can also find some example configuration files in the [example](./conf) folder.

You can also use CLI arguments to configure the program, like this:

```bash
python main.py conf_filepath=some.yaml optimizer.lr=0.05
```

Thus, the setting `optimizer.lr` in `some.yaml` will be overridden with the CLI input (i.e. 0.05).

## Inspiration & Contribution

If you find any bugs in my code or have any ideas to improve the quantization results, please feel free to open an issue. I will be glad to join the discussion.

NeuralZip originates from my another project, [an implementation of LSQ-Net](https://github.com/zhutmost/lsq-net).

It is powered by [PyTorch](https://pytorch.org), [PyTorch-Lightning](https://www.pytorchlightning.ai) and many other open-source projects.
Thanks for their excellent jobs.
