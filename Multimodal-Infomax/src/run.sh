
#-----------------------------MOSI 30%L=0---------------------------------
## TRAIN  
# rm -r /home/yingting/.cache/huggingface/transformers/transformers
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method missing --train_changed_modal language --train_changed_pct 0.3

# ## TEST
# ### Origin
# CUDA_VISIBLE_DEVICES=1 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method missing --train_changed_modal language --train_changed_pct 0.3 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
# ### 30%L = 0
# python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method missing --train_changed_modal language --train_changed_pct 0.3 --test_method missing --test_changed_modal language --test_changed_pct 0.3 --is_test      
# ### 30%L = N
# python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method missing --train_changed_modal language --train_changed_pct 0.3 --test_method g_noise --test_changed_modal language --test_changed_pct 0.3 --is_test

#-----------------------------MOSI 30%L=N---------------------------------
## TRAIN  
# rm -r /home/yingting/.cache/huggingface/transformers/transformers
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method g_noise --train_changed_modal language --train_changed_pct 0.3

# ## TEST 
# ### Origin
# CUDA_VISIBLE_DEVICES=1 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method g_noise --train_changed_modal language --train_changed_pct 0.3 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
# ### 30%L = 0
# python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method g_noise --train_changed_modal language --train_changed_pct 0.3 --test_method missing --test_changed_modal language --test_changed_pct 0.3 --is_test
# ### 30%L = N
# python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method g_noise --train_changed_modal language --train_changed_pct 0.3 --test_method g_noise --test_changed_modal language --test_changed_pct 0.3 --is_test

#-----------------------------MOSI 15%L=H---------------------------------
## TRAIN  
# rm -r /home/yingting/.cache/huggingface/transformers/transformers
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.15
# # TEST 
# # Origin
# CUDA_VISIBLE_DEVICES=1 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.15 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
# # 30%L = 0
# CUDA_VISIBLE_DEVICES=0 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.15 --test_method missing --test_changed_modal language --test_changed_pct 0.3 --is_test
# # 30%L = N
# CUDA_VISIBLE_DEVICES=0 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.15 --test_method g_noise --test_changed_modal language --test_changed_pct 0.3 --is_test


#-----------------------------MOSEI 30%L=0---------------------------------
# TRAIN
# rm -r /home/yingting/.cache/huggingface/transformers/transformers
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method missing --train_changed_modal language --train_changed_pct 0.3

# # TEST 
# # Origin
# CUDA_VISIBLE_DEVICES=1 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method missing --train_changed_modal language --train_changed_pct 0.3 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
# # 30%L = 0
# python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method missing --train_changed_modal language --train_changed_pct 0.3 --test_method missing --test_changed_modal language --test_changed_pct 0.3 --is_test
# # 30%L = N
# python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method missing --train_changed_modal language --train_changed_pct 0.3 --test_method g_noise --test_changed_modal language --test_changed_pct 0.3 --is_test

#-----------------------------MOSEI 30%L=N---------------------------------
# rm -r /home/yingting/.cache/huggingface/transformers/transformers
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method g_noise --train_changed_modal language --train_changed_pct 0.3

# # TEST 
# # Origin
# CUDA_VISIBLE_DEVICES=1 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method g_noise --train_changed_modal language --train_changed_pct 0.3 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
# # 30%L = 0
# python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method g_noise --train_changed_modal language --train_changed_pct 0.3 --test_method missing --test_changed_modal language --test_changed_pct 0.3 --is_test
# # 30%L = N
# python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method g_noise --train_changed_modal language --train_changed_pct 0.3 --test_method g_noise --test_changed_modal language --test_changed_pct 0.3 --is_test


#-----------------------------MOSEI 15%L=H---------------------------------
## TRAIN
# rm -r /home/yingting/.cache/huggingface/transformers/transformers
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.15
# # TEST 
# # Origin
# CUDA_VISIBLE_DEVICES=0 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.15 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
# # 30%L = 0
# CUDA_VISIBLE_DEVICES=0 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.15 --test_method missing --test_changed_modal language --test_changed_pct 0.3 --is_test
# # 30%L = N
# CUDA_VISIBLE_DEVICES=0 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.15 --test_method g_noise --test_changed_modal language --test_changed_pct 0.3 --is_test

#-----------------------------MOSI 5%L=H---------------------------------
## TRAIN  
printf "----------------------------------------------------------000000000"
rm -r /home/yingting/.cache/huggingface/transformers/transformers
CUDA_VISIBLE_DEVICES=3 python main.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.05

# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.05 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.05 --test_method missing --test_changed_modal language --test_changed_pct 0.3 --is_test
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.05 --test_method g_noise --test_changed_modal language --test_changed_pct 0.3 --is_test


#-----------------------------MOSI 10%L=H---------------------------------
## TRAIN  
printf "----------------------------------------------------------11111111111"
rm -r /home/yingting/.cache/huggingface/transformers/transformers
CUDA_VISIBLE_DEVICES=3 python main.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.1
# # TEST 
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.1 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.1 --test_method missing --test_changed_modal language --test_changed_pct 0.3 --is_test
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosi --contrast --lr_main 1e-3 --lr_mmilb 4e-3 --alpha 0.3 --beta 0.1 --batch_size 32 --d_vh 32 --d_ah 32 --train_method hybird --train_changed_modal language --train_changed_pct 0.1 --test_method g_noise --test_changed_modal language --test_changed_pct 0.3 --is_test



#-----------------------------MOSEI 5%L=H---------------------------------
## TRAIN
printf "----------------------------------------------------------222222222222"
rm -r /home/yingting/.cache/huggingface/transformers/transformers
CUDA_VISIBLE_DEVICES=2 python main.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.05

# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.05 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.05 --test_method missing --test_changed_modal language --test_changed_pct 0.3 --is_test
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.05 --test_method g_noise --test_changed_modal language --test_changed_pct 0.3 --is_test

#-----------------------------MOSEI 10%L=H---------------------------------
## TRAIN
printf "----------------------------------------------------------333333333333"
rm -r /home/yingting/.cache/huggingface/transformers/transformers
CUDA_VISIBLE_DEVICES=3 python main.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.1

# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.1 --test_method missing --test_changed_modal language --test_changed_pct 0 --is_test
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.1 --test_method missing --test_changed_modal language --test_changed_pct 0.3 --is_test
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset mosei --contrast --lr_main 5e-4 --lr_mmilb 1e-3 --alpha 0.1 --beta 0.05 --batch_size 64 --d_vh 64 --d_ah 16 --train_method hybird --train_changed_modal language --train_changed_pct 0.1 --test_method g_noise --test_changed_modal language --test_changed_pct 0.3 --is_test


printf "----------------------------------------------------------END---------"
