# Denoising diffusion models for protein structures 

Here we present a simple implementation of denoising diffusion models for protein structures. I created this code as an educational resource, to provide a simple, clean, and straightforward implementation that can be used to learn more about the use of diffusion models for protein structures, and provide a basis for further experimentation. 


## Code overview 

The code is organized into modules. The basic outline of the code organization is as follows:

```
protein-diffusion/
│
├── dompdb/          # Directory containing your PDB files
│   ├── 1abc
│   ├── 2def
│   └── ...
├── dataloader.py    # Handles data loading and preprocessing
├── model.py         # Contains model definitions 
├── training.py      # Handles the training loop
├── evaluation.py    # Contains evaluation metrics and methods
└── train.py         # Main script to train and evaluate the model
```

### Diffusion process and denoising models  

In the `model.py` file, you'll see the implementation of the diffusion process as well as the different denoising models. Currently, we have a few simple models implemented 

- `DenoisingDiffusionModel` which is similar to models used to denoise 3D point clouds and is not expected to perform particularly well on this task 
- `RegularTransformer` which is a plain transformer model (same caveat, these are baseline models that we are going to demonstrate improvement over)
- `MysteryTransformer` which is a simple SE(3)-equivariant transformer implementation that should perform better than the other models 


### Data representation 

To keep things very simple and educational, we start with a representation using only the alpha carbons from the structure 

- Additionally, we provide a `AlphaFoldBackboneRepresentation` module which represents backbones using all the backbone atoms, where a coordinate system is used for each individual residue 


## Experiments 

**Please note, this is an active project and a work in progress.** 


### Baseline models 

To begin, we train some baseline models. For a dataset, we'll use the CATH nonredundnat domain dataset that [we have used previously for protein design with graph attention networks](https://github.com/dacarlin/gato)


### SE(3) equivariant models 

[Coming soon!]