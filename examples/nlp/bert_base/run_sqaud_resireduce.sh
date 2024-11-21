#!/usr/bin/env bash

# Copyright (c) 2019-2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# scp -r 

echo " cd /ResiReduce/examples/nlp/bert_base "
cd /ResiReduce/examples/nlp/bert_base


echo "Container nvidia build = " $NVIDIA_BUILD_ID

export DIR_Model="/data/dataset/nlp/bert/pre-model/bert-base-uncased/uncased_L-12_H-768_A-12"
export DIR_DataSet="/data/dataset/nlp/bert"


percent=${1:-"0"}
compressor=${2:-"topk"}
# compressor=${2:-"none"}

init_checkpoint=${3:-"$DIR_Model/bert_base_wiki.pt"}
epochs=${4:-"4"}
batch_size=${5:-"4"}
learning_rate=${6:-"3e-5"}
warmup_proportion=${7:-"0.1"}
precision=${8:-"fp16"}
num_gpu=${9:-"8"}
seed=${10:-"1"}
squad_dir=${11:-"$DIR_DataSet/squad"}
vocab_file=${12:-"$DIR_Model/vocab.txt"}
OUT_DIR=${13:-"./bert_base_output"}
mode=${14:-"train eval"}
CONFIG_FILE=${15:-"$DIR_Model/bert_config.json"}
max_steps=${16:-"-1"}


##### ResiReduce-(w+2d)
memory=${17:-"resireducebert"}
# memory=${17:-"residual"}


# memory=${17:-"dimcprs"}
# memory=${17:-"dimcprs2d"}
# memory=${17:-"residualbertavgsig"}

##### ResiReduce-w
# memory=${17:-"residualbertweight"}
##### ResiReduce-a
# memory=${17:-"residualbertavg"}


# memory=${17:-"none"}

density="${density:-0.01}"


# compressor=${1:-"topk"}
# init_checkpoint=${2:-"$DIR_Model/bert_base_wiki.pt"}
# epochs=${3:-"2.0"}
# batch_size=${4:-"4"}
# learning_rate=${5:-"3e-5"}
# warmup_proportion=${6:-"0.1"}
# precision=${7:-"fp16"}
# num_gpu=${8:-"8"}
# seed=${9:-"1"}
# squad_dir=${10:-"$DIR_DataSet/squad"}
# vocab_file=${11:-"$DIR_Model/vocab.txt"}
# OUT_DIR=${12:-"./bert_base_output"}
# mode=${13:-"train eval"}
# CONFIG_FILE=${14:-"$DIR_Model/bert_config.json"}
# max_steps=${15:-"-1"}
# percent=${16:-"0"}


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  # mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
  # mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi



CMD=" horovodrun -np 8 -H n18:2,n15:2,n16:2,n19:2 python ./pytorch/run_squad_resireduce.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
  CMD+="--compressor=$compressor "
  CMD+="--density=$density "
  CMD+="--percent=$percent "
  CMD+="--memory=$memory "
fi

CMD+=" --do_lower_case "
# CMD+=" --bert_model=bert-large-uncased "
CMD+=" --bert_model=bert-base-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
# CMD+=" $use_fp16"

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE
