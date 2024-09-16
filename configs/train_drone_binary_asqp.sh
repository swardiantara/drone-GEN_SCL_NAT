python source/gen_scl_nat_main.py \
   --task asqp \
   --do_train \
   --do_direct_eval \
   --embedding sbert \
   --dataset acos_drone_binary \
   --model_name_or_path t5-base \
   --output_folder drone_sbert \
   --n_gpu 0 \
   --accelerator cpu \
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
   --model_prefix drone_binary_output