rm -r /home/yingting/.cache/huggingface/transformers/transformers
CUDA_VISIBLE_DEVICES=1 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method missing --train_changed_modal language --train_changed_pct 0.3 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
