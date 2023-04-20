# When Spiking neural networks meet temporal attention image decoding and adaptive spiking neuron

official implementation of  When Spiking neural networks meet temporal attention image decoding and adaptive spiking neuron

Accepted to ICLR Tiny 2023 (oral)!!

paper: https://openreview.net/forum?id=MuOFB0LQKcy
# Get started
install dependencies
~~~
pip install -r requirements.txt
~~~
initialize the fid stats
~~~
python init_fid_stats.py
~~~
# Training ANN VAE
As a comparison method, we prepared vanilla VAEs of the same network architecture built with ANN, and trained on the same settings.
~~~
python main_ann_vae exp_name -dataset dataset_name
~~~
args:

1.name: [required] experiment name <br>
2.dataset:[required] dataset name [mnist, fashion, celeba, cifar10] <br>
3.batch_size: default 250 <br>
4.latent_dim: default 128 <br>
5.checkpoint: checkpoint path (if use pretrained model) <br>
6.device: device id of gpu, default 0 <br>

If you find ALIF and TAID  module useful in your work, please cite the following source:
~~~
@misc{
qiu2023when,
title={When Spiking Neural Networks Meet Temporal Attention Image Decoding and Adaptive Spiking Neuron},
author={Xuerui Qiu and Zheng Luan and Zhaorui Wang and Rui-Jie Zhu},
year={2023},
url={https://openreview.net/forum?id=MuOFB0LQKcy}
}
~~~
