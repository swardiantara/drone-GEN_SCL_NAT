datasets=( acos_drone_binary acos_drone_multi acos_laptop_data acos_restaurant_data )
scenarios=( t5 )
tasks=( asqp gen_scl_nat )
seeds=( 14298463 246773155 30288239 82511865 90995999 )
for dataset in "${datasets[@]}"; do
    for scenario in "${scenarios[@]}"; do
        for task in "${tasks[@]}"; do
            for seed in "${seeds[@]}"; do
                python source/gen_scl_nat_main.py \
                    --task "$task" \
                    --do_train \
                    --do_direct_eval \
                    --scenario "$scenario" \
                    --dataset "$dataset" \
                    --model_name_or_path t5-base \
                    --output_folder train_outputs_scl \
                    --n_gpu 1 \
                    --accelerator gpu \
                    --train_batch_size 16 \
                    --eval_batch_size 16 \
                    --learning_rate 9e-5 \
                    --gradient_accumulation_steps 1 \
                    --num_train_epochs 50 \
                    --num_beams 5 \
                    --weight_decay 0.0 \
                    --seed "$seed" \
                    --cont_loss 0.05 \
                    --cont_temp 0.25 \
                    --model_prefix drone_binary_asqp \
                    --constrained_decoding 
            done
        done
    done
done