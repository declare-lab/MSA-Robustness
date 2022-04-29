# MSA-Robustness
NAACL 2022 paper on Analyzing Modality Robustness in Multimodal Sentiment Analysis

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
3. ```bash run.sh```


