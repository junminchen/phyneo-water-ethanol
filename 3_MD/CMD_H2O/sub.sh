#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=cmdNPTH2O
#SBATCH -N 1 -n 1 -t 800:00:00 -c 2
#SBATCH --gres=gpu:1 -p rtx3090
#SBATCH -o out -e err

module load cuda/11.8
module load cudnn/8.7-cuda_11.x
source activate /share/home/zhangrui/.conda/envs/dmff_sGNN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

jobname=$1   #${pdbfile%.*}_init.pdb   #02molL_init.pdb

#python gen_md_pdb.py $pdbfile
scrdir=/tmp

# clean folder
rm $scrdir/ipi_unix_dmff_*
echo "***** start time *****"
date

cd "$SCRIPT_DIR"
# run server
bash run_server.sh ${jobname} &
sleep 8

# check socket
ls -l $scrdir

# run client
bash run_client_dmff.sh ${jobname} & 
wait

echo "***** finish time *****"
date

sleep 1
 
