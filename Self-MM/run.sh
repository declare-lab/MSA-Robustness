####TRAIN
python run.py --modelName self_mm \
              --datasetName mosi \
              --train_method missing \
              --train_changed_modal language \
              --train_changed_pct 0.3
###TEST
# python test.py --modelName self_mm \
#                --datasetName mosi \
#                --train_method missing \
#                --train_changed_modal language \
#                --train_changed_pct 0.3 \
#                --test_method missing \
#                --test_changed_modal language \
#                --test_changed_pct 0 \
#                --is_test