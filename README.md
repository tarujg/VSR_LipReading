
# Visual Speech Recognition: Automated Lip Reading

## Authors: Utkrisht Rajkumar, Subrato Chakravorty, Taruj Goyal, Kaustav Datta

### Description

In this project, we experiment with various architectures (e.g. DenseNet+BGRU, DenseNet+Transformer, DenseNet+TCN, ConvNet+Transformer, etc) to determine the best end-to-end model for automated lip reading. We compare against LipNet based on this [pytorch-implementation](https://github.com/Fengdalu/LipNet-PyTorch). Our training procedures, data extraction and preprocessing, and evaluation scripts are mainly adapted from [Fengdalu/LipNet-PyTorch](Fengdalu/LipNet-PyTorch). 

## Table of contents

- [Requirements](#requirements)
- [Preprorcessing](#preprocessing)
- [How to run](#run)
- [Description of files](#description)

## Requirements <a name="requirements"></a>

## Preprocessing <a name="preprocessing"></a>

Link of processed lip images and text: 

BaiduYun: 链接:https://pan.baidu.com/s/1I51Xf-DzP1UgrXF-S0L5tg  密码:jf0l
Google Drive: https://drive.google.com/drive/folders/1Wn2EJw2101nF59eNDXEto6qXqfgDDucL

Download all parts and concatenate the files using the command 

```
cat GRID_LIP_160x80_TXT.zip.* > GRID_LIP_160x80_TXT.zip
unzip GRID_LIP_160x80_TXT.zip
rm GRID_LIP_160x80_TXT.zip
```

## How to run  <a name="run"></a>

Please contact urajkuma [at] eng.ucsd.edu for pretrained weights.

[FILL THIS OUT]

## Description of files  <a name="description"></a>
 

file name | Description of file 
--- | ---
eval.py | ...
[FILL OUT THE REST OF THESE THINGS]

------------------------------------------------------------------------

folder name | Description of file 
--- | ---
modules | ...
data | ...
scripts | ...
[FILL THIS OUT]
