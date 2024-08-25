# Decision Transformers
A project that applies decision transformers for the task of automated parameter selection in PnP-ADMM algorithm. The work focuses on the compressed sensing magnetic resonance imaging (CSMRI) inverse imaging problem.

In this work we have conducted two experiments:
1. Decision transformer for more flexible action generation in order to generate parameters to achieve a desired reward instruction (PSNR increment), specified by the initial RTG parameter, $R_0$, 
2. Decision transformer for optimal image recovery. Since we have trained the algorithm on a single policy the ability of the transformer to stich trajectories together is limited. Moreover, in this experiment we take parameters outputted by the transformer to define a distribution over possible actions, and the use an heuristic search algorithm to explore at inference. NoRef-IQA ARNIQA used to evaluate value of nodes.


## Training the model

To train the transformer using a single GPU the first step is to download the training data. For access to the training data message me at **joesharratt29@gmail.com** You can either train the model using a single GPU  or distributed training. The shell command on a single GPU is

```
python main.py --block_size 18 train --batch_size 48 --compile --save_every 1 --max_epochs 5
```


To train across multiple GPUs the command is as following:
```
python main.py --block_size 18 train --batch_size 48 --ddp --compile --save_every 1 --max_epochs 5
```

# Evaluation 
We have conducted two experiments, thus there are two models to evaluate they both can be found [here](https://1drv.ms/f/s!AvB5JAwBI_ZLbn_T9Y2DzVubO5c?e=EbDPUh) and are named appropriately. Place them in a `checkpoints` folder. To download the evaluation data follow this [link](https://1drv.ms/f/s!AvB5JAwBI_ZLgUE418feZOYQd_lI?e=EbFs1o) Put this data in a `evaluation/image_dir` folder. It is also necessary to download the weights to U-NET denoiser used as regulariser in the PnP algorithm which can be downloaded from [here](https://1drv.ms/f/s!AvB5JAwBI_ZLgSfr4VRXJgxRL2wJ?e=SoguCL)

To evaluate experiment 1, run the following command:
```
python main.py --block_size 18 --n_embeds 6 flex --max_timesteps 30
```

For the second experiment  the DT without tree search can be run as following:
```
python main.py --block_size 18 --n_embeds 9 eval --rtg 10 --max_timesteps 30
```

With tree search the transformer can be ran using the below command:
```
python main.py --block_size 18 --n_embeds 9 mcts --rtg 5 --max_timesteps 30 
```


