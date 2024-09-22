python source/train_embedding.py \
    --output_dir embeddings \
    --model_name_or_path bert-base-cased \
    --dataset acos_drone_binary \
    --label_name ac-sp \
    --strategy multi \
    --stage 2 \
    --margin 0.5 \
    --num_epochs 2 \
    --batch_size 64 \
    --exclude_duplicate_negative