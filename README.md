# 扬州大学毕业设计（论文）
## 陆新蓬（211301216 软件2102）


## Dependencies

- Python 3.7
- Pytorch 1.11
- mujoco 2.10
- d4rl
- wandb



## Quick Start

- Please refer to [command_parser.py](./configs/command_parser.py) for default hyper-parameters.
- near-expert policy $\pi^\ast$ checkpoints are provided in [offline_policy_checkpoints](./offline_policy_checkpoints) and obtained by using Cal-QL implemented in [CORL](https://github.com/tinkoff-ai/CORL).
- Av-PBC distilled datasets are available [here](https://drive.google.com/file/d/19yCQkCRy82YOqy8xZnOMqTDkgOhkAHWc/view?usp=sharing).

#### Syntheisze Behavioral Datasets

- **Av-PBC**

注意：--normalize会标准化，在迷宫环境下不加
```shell
python obd_bptt.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --q_weight --save_dir './saved_synset' --seed 0 --normalize
```

```shell
python obd_bptt.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --q_weight --save_dir './saved_synset_modified' --seed 0
```

```shell
 python obd_bptt.py --env 'antmaze-umaze-v2' --match_objective 'offline_data' --save_dir './saved_synset_ant' --seed 0 --synset_size 2048 --synset_init 'trajectory'
```

```shell
python obd_bptt.py --env 'antmaze-umaze-v2' --match_objective 'offline_policy' --q_weight --save_dir './saved_synset_antmaze' --seed 0 --orthogonal_init True --n_hidden_layers 5
```

```shell
 python sparse_reward.py --env 'antmaze-umaze-v2' --match_objective 'offline_data' --eval_freq 1000 --save_dir './saved_synset_ant' --group 'Evaluate' --seed 0 --synset_size 2048 --synset_init 'trajectory'
```

- **PBC**

```shell
python obd_bptt.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --save_dir './saved_synset' --seed 0
```

- **DBC**

```shell
python obd_bptt.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_data' --save_dir './saved_synset' --seed 0
```



#### Evaluate Behavioral Datasets

- **Standard evaluation**

```shell
python evaluate_synset.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --q_weight --eval_freq 1000 --save_dir './saved_synset' --group 'Evaluate' --seed 0 --normalize
```

在这里，可以修改合成数据集的保存地址

```shell
python synset_illustrate.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --q_weight --eval_freq 1000 --save_dir './saved_synset_modified' --group 'Evaluate' --seed 0
```

- **Ensemble evaluation**

```shell
python evaluate_synset.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --q_weight --eval_freq 1000 --eval_ensemble --ensemble_policy_num 10 --save_dir './saved_synset' --group 'Ensemble-Evaluate' -- --seed 0
```

#### Cross Arch/Optim Evaluation

```shell
python evaluate_cross_arch.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --q_weight --eval_freq 1000 --save_dir '/home/leaves/Data/OBD/q-value-weighted-synset' --group 'Cross-Arch-Optim-Evaluate' --seed 0
```
