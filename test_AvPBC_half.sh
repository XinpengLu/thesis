#!/bin/bash
SEED=(0)
ALPHA=0.005
SIZE=16
ALG='peav-pbc'
LEN_SEED=${#SEED[@]}

ENV=('halfcheetah-medium-replay-v2' 'halfcheetah-medium-expert-v2' 'halfcheetah-medium-v2')
LEN_ENV=${#ENV[@]}


for ((i=0; i<LEN_ENV; i++))
do
  for ((j=0; j<LEN_SEED; j++))
  do
	save_dir="./results/av_pbc/saved_synset_${ENV[i]}_seed_${SEED[j]}_alpha_${ALPHA}_size${SIZE}"
	python obd_bptt.py --env ${ENV[i]} --match_objective "offline_policy" --q_weight --save_dir "${save_dir}" --seed ${SEED[j]} --normalize --alpha ${ALPHA} --synset_size ${SIZE} --alg ${ALG} --project 'Size'
  done
done
wait
sleep 2


echo "Completed!"