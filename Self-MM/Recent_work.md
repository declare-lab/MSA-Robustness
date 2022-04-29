Analyzing Multimodal Sentiment Analysis

Predominant focus of recent works is towards imporve fusion, or alignment of modalities(text, audio, video) to boost overall performance - CMU-MOSI, MOSEI, MELD, etc.

1. MISA
2. Self-MM
3. Bi Bi-modal
4. Mutual Information Maximization
5. Cross-modal Transformers

https://github.com/declare-lab/multimodal-deep-learning

Overlooked aspects are: 1) Missing modality 2) Noise Sensitivity 3) Leveraging Strong Modalities

# Robustness Direction
1. RQ1) How robust are the present methods towards missing modalities? If not, is there an easy way to make them robust?
2. RQ2) How sensitive are modalities towards real-world noise? Are certain modalities more sensitive(e.g. Language modality)?

    Does Multimodal scores hamper the language unimodal score? If yes, can we reduce that.

# Improving Performance Direction
3. RQ3) Language operates as a strong modality in Multimodal Sentiment Analysis(MSA) datasets. But most works do not leverage on this fact and attempt to equally model all modalities. Can we focus on language as a dominant modality and the remaining audio and visual cues as complementary modalities?

    Takeaways: We are trying to leverage language modality - in a way that is robust.

# Revamping the benchmark
4. RQ4) Benchmarking MSA with modern neural feature extractors - Wav2vec for audio.(Engineering)

