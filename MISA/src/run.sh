####TRAIN
CUDA_VISIBLE_DEVICES=3 python train.py \
        --data mosi \
        --learning_rate 1e-5 \
        --optimizer RMSprop \
        --activation hardtanh \
        --train_method missing \
        --train_changed_modal language \
        --train_changed_pct 0.3

###TEST
# CUDA_VISIBLE_DEVICES=3 python test.py \
#         --data mosi \
#         --learning_rate 1e-5 \
#         --optimizer RMSprop \
#         --activation hardtanh \
#         --train_method missing \
#         --train_changed_modal language \
#         --train_changed_pct 0.3 \
#         --test_method missing \
#         --test_changed_modal language \
#         --test_changed_pct 0 \
#         --is_test

