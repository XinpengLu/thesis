#!/bin/bash
SEED=(0 1 2 3 4)
ALPHA=0
LEN_SEED=${#SEED[@]}

ENV=('halfcheetah-medium-replay-v2' 'halfcheetah-medium-expert-v2' 'halfcheetah-medium-v2')
LEN_ENV=${#ENV[@]}
CudaNum=($(eval echo {0..$((LEN_ENV-1))}))

for ((i=0; i<LEN_ENV; i++)) 
do
  for ((j=0; j<LEN_SEED; j++))
  do
	save_dir="./results/pbc/saved_synset_${ENV[i]}_seed_${SEED[j]}_alpha_${ALPHA}"
	CUDA_VISIBLE_DEVICES=${CudaNum[i]} python obd_bptt.py --env ${ENV[i]} --match_objective "offline_policy" --save_dir "${save_dir}" --seed ${SEED[j]} --normalize --alpha ${ALPHA} --n_iters 200000 &
  done
done
wait
sleep 2


echo "Completed!"