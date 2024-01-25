# DeepTunel

DeepTunel: Constructing Datasets and Training Deep Networks for the Direct Identification of Apoptosis Cells from H&E Staining Images

Accurate evaluation of apoptosis in tumor tissues is essential for the diagnosis and treatment of tumors. Terminal deoxynucleotidyl transferase dUTP nick end labeling (TUNEL) has been widely employed for detecting apoptotic cells in tissue sections for 30 years; however, its complex nature may yield erroneous results. Herein, a Deep Tunel network, capable of identifying and quantitatively analyzing apoptosis on H&E-stained tissue sections, is proposed, and its performance was assessed using 73 samples from four cancer models in three animal species (number of patches=44,189). The proposed network exhibited high accuracy (98.5%) in identifying apoptotic cells and was found to be 17.4% more precise than the current gold standard. Moreover, in comparison to pathologists, the proposed network exhibited a threefold increase in accuracy and was 100 times faster in detecting apoptotic cells. Thus, the proposed network may offer significant convenience and accuracy for investigating apoptosis in future studies.




## Install For Linux -->
## 1. Prepare data
Please refer to [datasets](./datasets/README.md)

## 2. Prepare environment

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name cvpods python=3.6 -y
conda activate cvpods
```

**Step 1.** Install corresponding version of torch and torchvision depends on the version of Cuda compilation tools you use.

Fisrt, check the version of Cuda compilation tools by:
```shell
nvcc -V
```
Assume your Cuda compilation tools vesion is 11.1,

If your machine is able to access the Internet, simply install using
```shell
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Or, you can install them offline, download torch and torchvision by the following urls:
[torch]https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp36-cp36m-linux_x86_64.whl
[torchvision]https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp36-cp36m-linux_x86_64.whl

if your platform is Windows, download torch and torchvision by the following urls:
[torch]https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp36-cp36m-win_amd64.whl
[torchvision]https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp36-cp36m-win_amd64.whl

Upload these two whl files to your offline machine, and install them by:
```shell
pip install /path/torch-1.8.0%2Bcu111-cp36-cp36m-linux_x86_64.whl
pip install /path/torchvision-0.9.0%2Bcu111-cp36-cp36m-linux_x86_64.whl
```
**Step 2.** Install other needed packages
```shell
# (for Windows)conda install libpython m2w64-toolchain -c msys2
pip install -r requirements.txt
```

**Step 3.**  Build cvpods as follows:
```shell
cd /path/DeepTunel
pip install -e .
```

## 4. Training
You can train our mothod DeepTunel on the downloaded datasets by the following command:
```shell
bash tools/train.sh
```

## 5. Custom datasets
If you want to train our method DeepTunel on your custom datasets, you need to do some extra jobs depending on your custom datasets as follows:

We takes datasets [MIDOG2022](https://midog2022.grand-challenge.org/) as an example.
We have MIDOG [images_path](/path/MIDOG2022/images), [annotations_json](/path/MIDOG2022/MIDOG2022_training_patch.json).

link MIDOG dataset to DeepTunel root by:
```shell
ln -s "/your-path/MIDOG2022/" "/DeepTunel/datasets/MIDOG2022"
```








