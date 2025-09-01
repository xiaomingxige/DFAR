
# *DFAR*
The PyTorch implementation for the DFAR: Deformable Feature Alignment and Refinement for moving infrared small target detection.
## 1. Pre-request
### 1.1. Environment
```bash
conda create -n DFAR python=3.10.11
conda activate DFAR
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

git clone --depth=1 https://github.com/xiaomingxige/DFAR
cd DFAR
pip install -r requirements.txt
```
### 1.2. DCNv2
#### Build DCNv2

```bash
cd nets/ops/dcn/
# You may need to modify the paths of cuda before compiling.
bash build.sh
```
#### Check if DCNv2 works (optional)

```bash
python simple_check.py
```
> The DCNv2 source files here is different from the [open-sourced version](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) due to incompatibility. [[issue]](https://github.com/open-mmlab/mmediting/issues/84#issuecomment-644974315)

### 1.3. Datasets
Our experiments are conducted on two datasets: **DAUB** and **IRDST**.
We would like to thank [SSTNet](https://github.com/UESTC-nnLab/SSTNet) for providing the datasets download links:
- **DAUB**: [Download Link](https://pan.baidu.com/s/1nNTvjgDaEAQU7tqQjPZGrw?pwd=saew) (Extraction Code: saew)
- **IRDST**: [Download Link](https://pan.baidu.com/s/1igjIT30uqfCKjLbmsMfoFw?pwd=rrnr) (Extraction Code: rrnr)
## 2. Train
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py > nohup.out &
```
## 3. Test
We utilize 1 NVIDIA GeForce RTX 3090 GPU for testing：

```bash
python vid_map_coco.py
```
## 4. Visualization
```bash
python vid_predict.py
```

## Citation
If you find this project is useful for your research, please cite:

```bash
@article{LUO2025111894,
  title={Deformable Feature Alignment and Refinement for moving infrared small target detection},
  author={Luo, Dengyan and Xiang, Yanping and Wang, Hu and Ji, Luping and Li, Shuai and Ye, Mao},
  journal={Pattern Recognition},
  pages={111894},
  year={2025},
  publisher={Elsevier}
}
```


## Acknowledgements
This work is based on [SSTNet](https://github.com/UESTC-nnLab/SSTNet) and [STDF-Pytoch](https://github.com/ryanxingql/stdf-pytorch). Thank [UESTC-nnLab](https://github.com/UESTC-nnLab) and [RyanXingQL](https://github.com/RyanXingQL) for sharing the codes.