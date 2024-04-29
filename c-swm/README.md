### Generate 3 body dataset

```
python data_gen/physics.py --num-episodes 5000 --history_length 11 --total_sequence_length 12 --fname data/balls_train.h5 --seed 1 
```

### Generate 2D Shapes dataset

```
python data_gen/env.py --env_id ShapesTrain-v0 --fname data/shapes_train.h5 --num_episodes 10 --seed 1 --history_length 1
```