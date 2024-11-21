# scp -r 

echo " cd /ResiReduce/examples/nlp/bert_large "
cd /ResiReduce/examples/nlp/bert_large


export DIR_Model="/data/dataset/nlp/bert/pre-model/bert-large-uncased/uncased_L-24_H-1024_A-16"
export DIR_DataSet="/data/dataset/nlp/bert"


init_checkpoint=${1:-"$DIR_Model/bert_large_pretrained_amp.pt"}
epochs=${2:-"4"}
batch_size=${3:-"4"}
learning_rate=${4:-"3e-5"}
warmup_proportion=${5:-"0.1"}
precision=${6:-"fp16"}
num_gpu=${7:-"8"}
seed=${8:-"1"}
squad_dir=${9:-"$DIR_DataSet/squad"}
vocab_file=${10:-"$DIR_Model/vocab.txt"}
# 输出模型和预测结果
OUT_DIR=${11:-"./squad_large"}


# train+eval
mode=${12:-"train eval"}
# mode=${12:-"train"}
CONFIG_FILE=${13:-"$DIR_Model/bert_config.json"}
max_steps=${14:-"-1"}


# setup
density="${density:-0.001}"
compressor="${compressor:-topk}"


##### ResiReduce-(w+2d)
memory=${17:-"resireducebert"}

##### ResiReduce-w
# memory=${17:-"residualbertweight"}
##### ResiReduce-a
# memory=${17:-"residualbertavg"}

# memory="${memory:-none}"
# memory="${memory:-residual}"
# memory="${memory:-dimcprs}"
# memory=${17:-"dimcprs2d"}

percent="${percent:-0}"


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

# CMD="python  $mpi_command ../run_squad_hvd.py "
CMD="HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 "
CMD=" horovodrun -np 8 -H n18:2,n15:2,n16:2,n19:2  python ./pytorch/run_squad_large_resireduce.py "
CMD+="--init_checkpoint=$init_checkpoint "
CMD+="--density=$density "
CMD+="--compressor=$compressor  "
CMD+="--memory=$memory "
CMD+="--percent=$percent "


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
fi

CMD+=" --do_lower_case "
# CMD+=" --bert_model=bert-large-uncased "
CMD+=" --bert_model=bert-large-uncased "
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
