# scp -r 

echo " cd /ResiReduce/examples/nlp/gpt2 "
cd /ResiReduce/examples/nlp/gpt2


OUT_DIR=${OUT_DIR:-"./log"}
DATA_DIR=${DATA_DIR:-"/data/dataset/nlp/openai-community/wikitext-2-raw-v1"}
# DATA_DIR=${DATA_DIR:-"/data/dataset/nlp/openai-community/wikitext-103-raw-v1"}
# DATA_DIR=${DATA_DIR:-"/data/dataset/nlp/openai-community/Wikipedia-cn"}
# DATA_DIR=${DATA_DIR:-"/data/dataset/nlp/openai-community/Chinese_modern_classical"}
# DATA_DIR=${DATA_DIR:-"/data/dataset/nlp/openai-community/OESD"}


# MODEL_DIR=${MODEL_DIR:-"/data/dataset/nlp/openai-community/gpt2"}
MODEL_DIR=${MODEL_DIR:-"/data/dataset/nlp/openai-community/gpt2-medium"}
# MODEL_DIR=${MODEL_DIR:-"/data/dataset/nlp/openai-community/gpt2-large"}



epochs="${epochs:-2}"
density="${density:-0.01}"
compressor="${compressor:-topk}"
# compressor="${compressor:-qsgd}"
# compressor="${compressor:-dgc}"

# compressor="${compressor:-none}"



# memory=${memory:-"none"}

##### ResiReduce-(w+2d) for GPT2
memory=${memory:-"resireducegpt2"}

####### ResiReduce-2d
# memory="${memory:-dimcprs2d}"

####### ResiReduce-a
# memory="${memory:-residualgpt2savg}"
####### ResiReduce-w
# memory="${memory:-residualgpt2savgweight}"

####### ResiReduce-d
# memory="${memory:-dimcprs}"



percent="${percent:-0}"

# batch_size="${batch_size:-8}"
batch_size="${batch_size:-2}"


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

# 
# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 1 -H node19:1  python run_clm_no_trainer_hvd.py --dataset_name /data/dataset/nlp/openai-community/wikitext-2-raw-v1 --dataset_config_name default --model_name_or_path /data/dataset/nlp/openai-community/gpt2 --output_dir  ./tmp/test-clm
# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H node15:1,node16:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1   python run_clm_no_trainer.py --dataset_name /data/dataset/nlp/openai-community/wikitext-2-raw-v1 --dataset_config_name default --model_name_or_path /data/dataset/nlp/openai-community/gpt2 --output_dir  ./tmp/test-clm
# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun  -np  2 -H  node19:1,node20:1   python run_clm_no_trainer_hvd.py  --dataset_name /data/dataset/nlp/openai-community/wikitext-2-raw-v1 --dataset_config_name default --model_name_or_path /data/dataset/nlp/openai-community/gpt2 --output_dir  ./tmp/test-clm
# 
# 


CMD=" HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 "
CMD=" horovodrun  -np  8 -H  n18:2,n19:2,n15:2,n16:2   python ./run_gpt2_resireduce.py   "

CMD+=" --dataset_name $DATA_DIR --dataset_config_name default  "
CMD+=" --model_name_or_path $MODEL_DIR "
CMD+=" --num_train_epochs=$epochs  "
CMD+=" --per_device_train_batch_size=$batch_size  --per_device_eval_batch_size=$batch_size  "
CMD+=" --density=$density --compressor=$compressor --memory=$memory --percent=$percent "

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE




