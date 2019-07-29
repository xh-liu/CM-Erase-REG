# Improving Referring Expression Grounding with Cross-modal Attention-guided Erasing

Code for the CVPR 2019 Paper [Improving Referring Expression Grounding with Cross-modal Attention-guided Erasing](https://arxiv.org/pdf/1903.00839.pdf)

## Prerequisites

* Python 2.7
* Pytorch 0.3.0
* CUDA 8.0

## Installation

1. Clone the CM-Erase repository

```
git clone --recursive https://github.com/xh-liu/CM-Erase
```

2. Prepare the submodules and associated data

* Mask R-CNN: Follow the instructions of my [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn) repo, preparing everything needed for `pyutils/mask-faster-rcnn`.

* REFER API and data: Use the download links of [REFER](https://github.com/lichengunc/refer) and go to the foloder running `make`. Follow `data/README.md` to prepare images and refcoco/refcoco+/refcocog annotations.

* refer-parser2: Follow the instructions of [refer-parser2](https://github.com/lichengunc/refer-parser2) to extract the parsed expressions using [Vicente's R1-R7 attributes](http://tamaraberg.com/papers/referit.pdf). **Note** this sub-module is only used if you want to train the models by yourself.


## Training
1. Prepare the training and evaluation data by running `tools/prepro.py`:

```
python tools/prepro.py --dataset refcoco --splitBy unc
```

2. Download the Glove pretrained word embeddings at [Google Drive](https://drive.google.com/open?id=1U-vy9_mqyaVXfQMkynhTli8TMQgFYZfs).

3. Extract features using Mask R-CNN, where the `head_feats` are used in subject module training and `ann_feats` is used in relationship module training.

```bash
CUDA_VISIBLE_DEVICES=gpu_id python tools/extract_mrcn_head_feats.py --dataset refcoco --splitBy unc
CUDA_VISIBLE_DEVICES=gpu_id python tools/extract_mrcn_ann_feats.py --dataset refcoco --splitBy unc
```

4. Detect objects/masks and extract features (only needed if you want to evaluate the automatic comprehension). We empirically set the confidence threshold of Mask R-CNN as 0.65.

```bash
CUDA_VISIBLE_DEVICES=gpu_id python tools/run_detect.py --dataset refcoco --splitBy unc --conf_thresh 0.65
CUDA_VISIBLE_DEVICES=gpu_id python tools/run_detect_to_mask.py --dataset refcoco --splitBy unc
CUDA_VISIBLE_DEVICES=gpu_id python tools/extract_mrcn_det_feats.py --dataset refcoco --splitBy unc
```

5. Pretrain the network (CM-Att) with ground-truth annotation:

```bash
./experiments/scripts/train_mattnet.sh GPU_ID
```

6. Train the network with cross-modal erasing (CM-Att-Erase):

```bash
./experiments/scripts/train_erase.sh GPU_ID
```

## Evaluation

Evaluate the network with ground-truth annotation:

```bash
./experiments/scripts/eval_easy.sh GPU_ID
```

Evaluate the network with Mask R-CNN detection results:

```bash
./experiments/scripts/eval_dets.sh GPU_ID 
```

## Pre-trained Models

We provide the pre-trained models for RefCOCO, RefCOCO+ and RefCOCOg. Download them from [Google Drive](https://drive.google.com/open?id=1nYiDIU4nKOTz2dyOVQ7ThPkvlSeI_Nwp) and put them under `./output` folder.

## Citation

If you find our code useful for your research, please consider citing:
```
@inproceedings{liu2019improving,
  title={Improving Referring Expression Grounding with Cross-modal Attention-guided Erasing},
  author={Liu, Xihui and Wang, Zihao and Shao, Jing and Wang, Xiaogang and Li, Hongsheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1950--1959},
  year={2019}
}
```

## Acknowledgement 

This project is built on [Pytorch implementation](https://github.com/lichengunc/MAttNet) of [MAttNet: Modular Attention Network for Referring Expression Comprehension](https://arxiv.org/pdf/1801.08186.pdf) in [CVPR 2018](http://cvpr2018.thecvf.com/).

