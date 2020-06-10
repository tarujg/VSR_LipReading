
# Visual Speech Recognition: Automated Lip Reading

## Authors: Utkrisht Rajkumar, Subrato Chakravorty, Taruj Goyal, Kaustav Datta

### Description
In this project, we experiment with various architectures (e.g. DenseNet+BGRU, DenseNet+Transformer, DenseNet+TCN, ConvNet+Transformer, etc) to determine the best end-to-end model for automated lip reading. We compare against LipNet based on this [pytorch-implementation](https://github.com/Fengdalu/LipNet-PyTorch). Our training procedures, data extraction and preprocessing, and evaluation scripts are mainly adapted from [Fengdalu/LipNet-PyTorch](Fengdalu/LipNet-PyTorch). 



## Table of contents

- [Usage](#usage)
  - [Flags](#flags)
    - `-1`
    - `-a`   (or) `--all`
    - `-f`   (or) `--files`
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Datasets](#datasets)
- [Authors](#authors)
- [References](#references)
- [License](#license)

## Usage

How to train/evaluation script

### Flags

- With `-f` (or) `--files` : Shows only files
- With `-h` (or) `--help` : Prints a very helpful help menu

## Dependencies

* python 3.7
* pytorch 1.21
* opencv-python 3.4.0

## Installation 

1. Install PyTorch - version
2. Installation
    ```bash
    pip install -r requirements.txt
    ```

## Datasets

- LRW [link](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)
- LRS2


## Authors

* **Kaustav Datta** - kdatta@ucsd.edu
* **Subrato Chakravorty** - suchakra@ucsd.edu
* **Taruj Goyal** - tgoyal@ucsd.edu
* **Utkrisht Rajkumar** - urajkuma@ucsd.edu


## References

```
@article{petridis2018end,
  title={End-to-end audiovisual speech recognition},
  author={Petridis, Stavros and Stafylakis, Themos and Ma, Pingchuan and Cai, Feipeng and Tzimiropoulos, Georgios and Pantic, Maja},
  booktitle={ICASSP},
  pages={6548--6552},
  year={2018},
  organization={IEEE}
}
```

## License

The MIT License (MIT) 2020
