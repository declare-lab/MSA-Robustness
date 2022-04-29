#TRAIN
CUDA_VISIBLE_DEVICES=1 python main.py \
            --dataset mosi \
            --data_path ./datasets/MOSI \
            --use_bert \
            --lr 1e-4 \
            --train_method missing \
            --train_changed_modal language \
            --train_changed_pct 0.3

#TEST
# CUDA_VISIBLE_DEVICES=1 python test.py \
#             --dataset mosi \
#             --data_path ./datasets/MOSI \
#             --use_bert \
#             --lr 1e-4 \
#             --train_method missing \
#             --train_changed_modal language \
#             --train_changed_pct 0.3 \
#             --test_method missing \
#             --test_changed_modal language \
#             --test_changed_pct 0 \
#             --is_test