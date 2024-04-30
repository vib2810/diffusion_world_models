# Diffusion World Models
Course Project for Probabilistic Graphical Models (10-708) at CMU

Team Members:
- Vibhakar Mohta (vmohta@cs.cmu.edu)
- Praveen Venkatesh (pvenkat2@cs.cmu.edu)

## C-SWM Experiments
1. Build Docker:
```bash
docker build -t pgm_project .
```
2. Start docker by running 
```bash
./run_docker.sh
```

Now, the following experiments can be run:

### Generate 2D Shapes dataset

```
cd c-swm/
python data_gen/env.py --env_id ShapesTrain-v0 --fname data/shapes_train.h5 --num_episodes 1000 --seed 1 --history_length 1
python data_gen/env.py --env_id ShapesEval-v0 --fname data/shapes_eval.h5 --num_episodes 10000 --seed 1 --history_length 1
```


### Train 2D shapes models

```
#AE
python train.py --dataset data/shapes_train.h5 --name shapes --decoder
#cSWM
python train.py --dataset data/shapes_train.h5 --name shapes 
```


## Diffusion World Models Experiments

### Training VAE

First train the VAE. Ensure that you have generated the dataset as per the c-swm section above.

```
python train_vae.py --environment shapes
```


### Training LDM

To train the Latent Diffusion model, first change the config file to point to the correct VAE model to be loaded as the encoder decoder pair. Then, train the LDM using:

```
python train_diffusion.py --environment shapes --expt_name expt_name
```


Evaluation can be done by running:

```
python analysis/eval_diffusion.py
```

Ensure that you change the config to the correct model weight locations.

