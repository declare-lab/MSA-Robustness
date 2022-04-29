##TRAIN
CUDA_VISIBLE_DEVICES=0,1 python main.py \
                --dataset mosi \
                --batch_size 128 \
                --lr 1e-4 \
                --optim Adam \
                --num_heads 10 \
                --embed_dropout 0.3 \
                --attn_dropout 0.2 \
                --out_dropout 0.1 \
                --clip 0.8 \
                --num_epochs 100 \
                --aligned \
                --nlevels 4 \
                --use_bert \
                --train_method missing \
                --train_changed_modal language \
                --train_changed_pct 0.3
##TEST
# CUDA_VISIBLE_DEVICES=0,1 python main.py \
#                 --dataset mosi \
#                 --batch_size 128 \
#                 --lr 1e-4 \
#                 --optim Adam \
#                 --num_heads 10 \
#                 --embed_dropout 0.3 \
#                 --attn_dropout 0.2 \
#                 --out_dropout 0.1 \
#                 --clip 0.8 \
#                 --num_epochs 100 \
#                 --aligned \
#                 --nlevels 4 \
#                 --use_bert \
#                 --train_method missing \
#                 --train_changed_modal language \
#                 --train_changed_pct 0.3 \
#                 --test_method missing \
#                 --test_changed_modal language \
#                 --test_changed_pct 0 --is_test
