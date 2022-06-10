# MSA-Robustness
NAACL 2022 paper on [Analyzing Modality Robustness in Multimodal Sentiment Analysis](https://arxiv.org/pdf/2205.15465.pdf)

# Setup the environment
Configure the environment of different models respectively, configure the corresponding environment according to the requirements.txt in the model directory.

# Data Download
- Install [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). Ensure, you can perform ```from mmsdk import mmdatasdk```.  

# Running the code
Take MISA as an example

1. ```cd MISA```
2. ```cd src```
3. Set ```word_emb_path``` in ```config.py``` to [glove file](http://nlp.stanford.edu/data/glove.840B.300d.zip).
4. Set ```sdk_dir``` to the path of CMU-MultimodalSDK.
3. ```bash run.sh``` When doing robustness training, run the "TRAIN" section of run.sh, and when doing diagnostic tests, run the "TEST" section of run.sh.

&ensp;&ensp;&ensp;&ensp;```--train_method``` means the robustness training method, one of ```{missing, g_noise, hybird}```, ```missing``` means set to zero noise, ```g_noise``` means set to Gaussian Noise, ```hybird``` means the data of train_changed_pct is set to zero_noise, and the data of train_changed_pct is set to Gaussian_Noise.

&ensp;&ensp;&ensp;&ensp;```--train_changed_modal``` means the modality of change during training, one of ```{language, video, audio}```.

&ensp;&ensp;&ensp;&ensp;```--train_changed_pct``` means the percentage of change during training, can set between ```0~1```.

&ensp;&ensp;&ensp;&ensp;```--test_method``` means the diagnostic tests method, one of ```{missing, g_noise, hybird}```, ```missing``` means set to zero noise, ```g_noise``` means set to Gaussian Noise, ```hybird``` means the data of test_changed_pct is set to zero_noise, and the data of test_changed_pct is set to Gaussian_Noise.

&ensp;&ensp;&ensp;&ensp;```--test_changed_modal``` means the modality of change during testing, one of ```{language, video, audio}```.

&ensp;&ensp;&ensp;&ensp;```--train_changed_pct``` means the percentage of change during testing, can set between ```0~1```.

# Citation

```
@article{hazarika2022analyzing,
  title={Analyzing Modality Robustness in Multimodal Sentiment Analysis},
  author={Hazarika, Devamanyu and Li, Yingting and Cheng, Bo and Zhao, Shuai and Zimmermann, Roger and Poria, Soujanya},
  publisher={NAACL 2022},
  year={2022}
}

```


