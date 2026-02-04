# Diffusing DeBias: Synthetic Bias Amplification for Model Debiasing

# Abstract
The effectiveness of deep learning models in classification tasks is often challenged by the quality and quantity of training data whenever they are affected by strong spurious correlations between specific attributes and target labels. This results in a form of bias affecting training data, which typically leads to unrecoverable weak generalization in prediction. This paper addresses this problem by leveraging bias amplification with generated synthetic data only: we introduce Diffusing DeBias (DDB), a novel approach acting as a plug-in for common methods of unsupervised model debiasing, exploiting the inherent bias-learning tendency of diffusion models in data generation. Specifically, our approach adopts conditional diffusion models to generate synthetic bias-aligned images, which fully replace the original training set for learning an effective bias amplifier model to be subsequently incorporated into an end-to-end and a two-step unsupervised debiasing approach. By tackling the fundamental issue of bias-conflicting training samples’ memorization in learning auxiliary models, typical of this type of technique, our proposed method outperforms the current state-of-the-art in multiple benchmark datasets, demonstrating its potential as a versatile and effective tool for tackling bias in deep learning models.

## Getting Started
### Requirements
+ python 3.10+
+ pytorch 2.0+ (with torchvision)
+ An NVIDIA GPU


### Datasets
We implemented automatic download for the benchmark datasets analyzed in this study, therefore there is no need to manually add them. For the Urbancars and Imagenet9 datasets, please refer to [Whac-A-Mole](https://github.com/facebookresearch/Whac-A-Mole/blob/main/README.md#urbancars-experiments) and [ReBias](https://github.com/clovaai/rebias) repositories, respectively.


### Setup Python Environment:

To set up your python environment, you can use venv+pip and leverage the provided dependency file "requirements.txt":
  ```
  python3.10 -m venv <env_path>
  source <env_path>/bin/activate
  pip install -r requirements.txt
  ```


# Running DDB Experiments  
### Synthetic Image Generations

To run the Debiasing Recipes, place generated images in the directory ```Debiasing/data/synthetic```. Specifically, ```w_1/imagenet``` should contain the synthetic images used for the main results. Thus, before running the debiasing step you should have already the generated images at hand.

## Diffusing the Bias 
To run components from this part, you need to change your current working directory to ```DiffuseBias```, then you can launch both CDPM training and Image Generation as follows:
- Launch CDPM model training
  ```
  python runCDPM.py --state train --iterations 100000 --batch_size 32 --dataset waterbirds --img_size 64 --device cuda:0
  ```
- Generate synthetic images
  ```
  python runCDPM.py --state eval --load_weights path/to/checkpoint.pt --batch_size 100 --dataset waterbirds --img_size 64 --device cuda:0
  ```

Generated image captions, used for quantitatively validating identified biases, can be obtained by running:
  ```
      python captions_generator.py /path/to/synthetic/images.npy/directory/ --device cpu
  ```

## Debiasing Recipes
To run the different debiasing recipes you need to change your current working directory to ```Debiasing```, then create the directories ```outputs``` and ```saved_models```, finally launch Recipe I and Recipe II as follows:

### Recipe I: two-step debiasing

To execute DDB Recipe I with three different runs on different seeds, an example command is
```
bash scripts/waterbirds_seeds.sh
```

