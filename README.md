# SDGF 

This is an implementation of [D-CTNet: A Dual-Branch Channel-Temporal Forecasting Network with Frequency-Domain Correction].

## Usage

- Train and evaluate SDGF
  - You can use the following command:`sh ./scripts/ETTh1.sh`.

- Train your model
  - Add model file in the folder `./models/your_model.py`.
  - Add model in the ***class*** Exp_Main.

## Model

Our proposed SDGF Network consists of three key modules: Graph Structure Learning module that uses RevIN normalization and Multi-level Wavelet Decomposition to construct static and dynamic inter-series graphs, an Attention Gated Fusion module that adaptively integrates static and dynamic graph features, and Temporal Feature Learning module that employs multi-kernel dilated convolutions and an MLP-based output layer to capture temporal dependencies and generate predictions.

<div align=center>
<img src="https://github.com//SDGF/blob/main/pic/" width='45%'> <img src="https://github.com//SDGF/blob/main/pic/model2.jpg" width='47%'>
</div>


## Citation

If you find this repo useful, please cite our paper as follows:
```

```

## Contact
If you have any questions, please contact us or submit an issue.

## Acknowledgement

We appreciate the valuable contributions of the following GitHub.

- LTSF-Linear (https://github.com/cure-lab/LTSF-Linear)
- TimesNet (https://github.com/thuml/TimesNet)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- MSGNet (https://github.com/YoZhibo/MSGNet)
- MTGnn (https://github.com/nnzhan/MTGNN)