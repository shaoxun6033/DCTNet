# D-CTNet 

This is an implementation of [D-CTNet: A Dual-Branch Channel-Temporal Forecasting Network with Frequency-Domain Correction].

## Usage

- Train and evaluate D-CTNet
  - You can use the following command:`sh ./scripts/ETTh1.sh`.

- Train your model
  - Add model file in the folder `./models/your_model.py`.
  - Add model in the ***class*** Exp_Main.

## Model
Proposed a Patch-Based Dual-Branch Channel-Temporal Forecasting Network (D-CTNet). Particularly, with a parallel dual-branch design incorporating linear temporal modeling layer and channel attention mechanism, our method explicitly decouples and jointly learns intra-channel temporal evolution patterns and dynamic multivariate correlations. Furthermore, a global patch attention fusion module goes beyond the local window scope to model long range dependencies. Most importantly, aiming at non-stationarity, a Frequency-Domain Stationarity Correction mechanism adaptively suppresses distribution shift impacts from environment change by spectrum alignment.

<div align=center>
<img src="https://github.com/shaoxun6033/DCTNet/blob/main/pic/image.png" width='47%'>
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
