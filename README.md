# MHFusionNet: Multiple Hypotheses Fusion-Based Approach for 3D Human Pose Estimation

MHFusionNet is a 3D human pose estimation that utilizes a fusion-based approach. It aggregates multiple hypotheses generated from 2D-to-3D models and refines the final 3D pose through a dedicated Fusion Network. 

<p align="center">
  <img width="480" height="320" alt="image" src="https://github.com/user-attachments/assets/5d832e16-73c6-45a3-9401-633d66888262" />
</p>

We used a pre-trained multi-hypotheses model ([D3DP](https://github.com/paTRICK-swk/D3DP)) to generate multiple 3D human pose. Then, the Fusion Network (FN) was designed based on two strategies Feature Fusion (𝐹𝐹) and Early Fusion (𝐸𝐹) techniques.

<p align="center">
  <img width="480" height="320" alt="image" src="https://github.com/user-attachments/assets/0823a4f1-ca78-40a6-8c5b-9d71df7bae00" />
</p>

---

## 📦 Dependencies

Make sure the following packages are installed in your Python environment:

- Python 3.8.2  
- PyTorch >= 0.4.0  
- matplotlib==3.1.0  
- [einops](https://github.com/arogozhnikov/einops)  
- [timm](https://github.com/huggingface/pytorch-image-models)  
- tensorboard  

Install via pip:

```bash
pip install torch matplotlib==3.1.0 einops timm tensorboard
```
## Dataset: Human3.6M

We evaluate our model on the Human3.6M dataset. We use the same setup as the [D3DP](https://github.com/paTRICK-swk/D3DP) model. Please download the preprocessed data from the D3DP GitHub repository.

### Placement:

Place all dataset files in the ./dataset directory of this project.

```bash
MHFusionNet/
├── dataset/
│   ├── data_2d_h36m_gt.npz
│   ├── data_2d_h36m_cpn_ft_h36m_dbb.npz
│   └── data_3d_h36m.npz
```
## Model Checkpoint

The Fusion Network relies on multiple hypotheses as input. These hypotheses are generated using a pre-trained model (e.g., D3DP). To evaluate the Fusion Network, Place the Fusion Network checkpoint in the ./checkpoint/ directory. Ensure the multiple hypotheses data is also available (generated from D3DP).

```bash
MHFusionNet/
├── checkpoint/
│   └── fusion_model_best.pth
```

## Evaluating Our Model

```bash
python3 main_eval_v2.py -b 4 -gpu 0,1
```

or:

```bash
python3 train_fusion_v2.py --evaluation -gpu 0,1
```

## Training from scratch
Before training, ensure you have:

The dataset in the ./dataset directory

Multiple hypotheses data available in ./data_hypotheses for training the fusion model

To begin training:
```bash
python3 train_fusion_v2.py
```

