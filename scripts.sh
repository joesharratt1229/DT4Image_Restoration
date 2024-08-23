#train script
python main.py --block_size 18 train --batch_size 48 --compile --save_every 1 --max_epochs 5

#train ddp script
python main.py --block_size 18 train --batch_size 48 --ddp --compile --save_every 1 --max_epochs 5

#eval script
python main.py --block_size 18 --n_embeds 9 eval --rtg 10 --max_timesteps 30

#mcts script
python main.py --block_size 18 --n_embeds 9 mcts --rtg 5 --max_timesteps 30 

#fleixble script
python main.py --block_size 18 --n_embeds 6 flex --max_timesteps 30