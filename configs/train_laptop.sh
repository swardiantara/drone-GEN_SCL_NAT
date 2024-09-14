python source/gen_scl_nat_main.py \
   --task gen_scl_nat \
   --do_train \
   --do_direct_eval \
   --do_inference \
   --embedding sbert \
   --dataset acos_laptop_data \
   --model_name_or_path t5-base \
   --output_folder train_outputs \
   --n_gpu 1 \
   --train_batch_size 16 \
   --eval_batch_size 16 \
   --learning_rate 9e-5 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 2 \
   --num_beams 5 \
   --weight_decay 0.0 \
   --seed 123 \
   --cont_loss 0.05 \
   --cont_temp 0.25 \
   --model_prefix laptop_output