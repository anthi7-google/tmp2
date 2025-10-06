if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --mn)
      MN="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
DEFAULT_MODEL_NAME="CP_RMCN"

seq_len=336
pred_len=$seq_len
model_name=${MN:-$DEFAULT_MODEL_NAME}

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2024
for ch in 0 1 2 3 4 5 6
do
    python3 -u CP_run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 5\
      --lradj 'TST'\
      --pct_start 0.2\
      --channel $ch\
      --itr 1 --batch_size 128 --learning_rate 0.0001 \
      --resolution 'hour' \
      2>&1 | tee logs/ChannelPrediction/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$ch.log 
done