#!/bin/bash
SEED=(0 1 2 3 4)
ALPHA=0
SIZE=256
ALG='av-pbc'
LEN_SEED=${#SEED[@]}

ENV=('hopper-medium-replay-v2' 'hopper-medium-expert-v2' 'hopper-medium-v2')
LEN_ENV=${#ENV[@]}
CudaNum=($(eval echo {0..$((LEN_ENV-1))}))

for ((i=0; i<LEN_ENV; i++))
do
  for ((j=0; j<LEN_SEED; j++))
  do
		save_dir="./results/av_pbc/saved_synset_${ENV[i]}_seed_${SEED[j]}_alpha_${ALPHA}_size${SIZE}"
	  CUDA_VISIBLE_DEVICES=${CudaNum[i]} python evaluate_synset.py --env ${ENV[i]} --match_objective 'offline_policy' --q_weight --eval_freq 1000 --save_dir "${save_dir}" --group 'Evaluate' --seed ${SEED[j]} --normalize --alpha ${ALPHA} --n_iters 20000 --alg ${ALG} &
  done
done
wait
sleep 2


echo "Completed!"