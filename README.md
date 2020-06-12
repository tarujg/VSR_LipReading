
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
* PyTorch 1.0+
* opencv-python
* editdistance

[FILL THIS OUT]

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

Depending upon the type of frontend and backend you want to use, change the ```isDense``` and ```isTransformer``` parameter in main.py. Frontend will be DenseNet3D if ```isDense``` is ```True``` otherwise, ```STCNN``` is used. Similary, if ```isTransformer``` is ```True```, model will have a transformer backend else BiGRU is used.  

For example, to train a DenseNet3D + Transformer model on GRID Corpus.
set ```isDense=True``` and ```isTransformer=True```, and then run
```
python main.py
```


## Description of files  <a name="description"></a>
 

file name | Description of file 
--- | ---
main.py | Primary file to run, that trains the model
model.py | Builds the model, using the specified front end (STCNN/Dense3D) and backend (Transformer/GRU)
dataset.py | Creates the dataset from specified paths of the preprocessed images
options.py | Contains configuration parameters
eval.py | Evaluates a given model on a specified dataset, outputting the average wer and cer
eval2.py | Evaluates the wer and cer for a given sentence - given a predicted sentence, and a ground truth alignment file of a sentence from the GRID corpus
cvtransforms.py | Computes transformations on the data - horizontal flipping and color normalisation


------------------------------------------------------------------------

folder name | Description
--- | ---
modules | Contains model architectures for Dense3d, Transformer, and TCN
data | Contains paths to image folders for both overlap and unseen datasets (with both full data as well as a subset)
scripts | Contains a dataset download script download_gridcorpus.sh, and preprocessing scripts for face and lip extraction

