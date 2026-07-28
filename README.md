# Attention-Based Image Caption Generation (ABICG)

Generating natural-language captions for images using an InceptionV3 encoder, a GRU decoder, and Bahdanau attention — trained and evaluated on the Flickr8k dataset.

Published as: *Uday Kulkarni, Kushagra Tomar, Mayuri Kalmat, Rakshita Bandi, Pranav Jadhav, Meena S M* — "Attention Based Image Caption Generation (ABICG) using Encoder-Decoder Architecture", Dept. of CSE, KLE Technological University, Hubballi, India. Full paper: [ABICG_Final.pdf](ABICG_Final.pdf).

## Overview

Image captioning asks a model to look at a picture and describe it the way a person would — not just naming objects, but expressing how they relate to each other and what's happening in the scene. This project frames captioning as a classic encoder–decoder problem, adds an attention mechanism on top so the decoder can focus on different regions of the image at each generated word, and evaluates the result against the same architecture without attention.

**Headline result:** adding Bahdanau attention lifts BLEU-based accuracy from **79%** (plain InceptionV3-GRU) to **90%** (ABICG), and drives training loss down from 0.8647 to 0.0050 over 25 epochs.

## Architecture

```text
Image → InceptionV3 (Encoder) → tanh → Bahdanau Attention → GRU (Decoder) → Caption
                                              ↑
                                        Embedding(word_t-1)
```

- **Encoder — InceptionV3.** A CNN pretrained on ImageNet, with its final classification layer removed so it outputs a spatial feature map (8×8×2048) instead of class scores. Chosen over VGG-16/ResNet alternatives for having fewer parameters, so it's cheaper to compute and more memory-efficient without giving up accuracy.
- **Attention — Bahdanau (additive) attention.** At each decoding timestep, the previous decoder hidden state and the encoder's feature map are combined (`tanh(W1·h_d + W2·features)`), scored, and passed through a `softmax` to get per-region attention weights `α`. The context vector `c_vec = α · features` tells the decoder which parts of the image matter for the *next* word — this is what lets the model produce grounded captions instead of one generic description of the whole scene.
- **Decoder — GRU.** A Gated Recurrent Unit consumes the context vector plus the embedding of the previously generated word and produces the next word. GRU was preferred over LSTM/vanilla RNN here for its simpler two-gate structure (update + reset gates), which trains faster and is less prone to vanishing gradients than a vanilla RNN, while needing fewer parameters than LSTM.

See [ABICG_Final.pdf](ABICG_Final.pdf) for the full architecture diagrams and derivations (Sections III–IV).

## Dataset

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,091 images, each paired with 5 human-written captions (40,455 captions total).

Preprocessing pipeline:

- **Images:** resized to 299×299×3 and normalized to [-1, 1] to match InceptionV3's expected input.
- **Captions:** lowercased, stripped of punctuation and alphanumeric tokens, wrapped with `<start>`/`<end>` tokens, tokenized, and padded (post-padding) to the longest caption length. Vocabulary is capped at 5,000 words for memory efficiency; anything outside it maps to `<UNK>`. Masking is applied during training so padded timesteps don't contribute to the loss.
- **Split:** 80/20 train/test via `sklearn.train_test_split`.

## Repository contents

| File | Description |
| --- | --- |
| [ABICG-Model-Code.ipynb](ABICG-Model-Code.ipynb) | End-to-end notebook: data loading/EDA, preprocessing, InceptionV3 feature extraction, Encoder/Attention/Decoder model classes, training loop, BLEU evaluation, and attention-map visualization. |
| [ABICG_Final.pdf](ABICG_Final.pdf) / [ABICG_Final.docx](ABICG_Final.docx) | The paper describing the model, methodology, and results. |
| [Team_02_Report.pdf](Team_02_Report.pdf) / [Team-02 Report.zip](Team-02%20Report.zip) | Extended project report. |
| [ML Team 02.pptx](ML%20Team%2002.pptx) | Project presentation slides. |
| [ML- poster1 - Kushagra.pptx](ML-%20poster1%20-%20Kushagra.pptx), [ML- poster2 - Kushagra.pptx](ML-%20poster2%20-%20Kushagra.pptx) | Conference/academic poster decks. |

## Getting started

### Requirements

- Python 3.8+
- A GPU is strongly recommended — InceptionV3 feature extraction and GRU training are compute-heavy (the original run took ~2h13m on an NVIDIA GTX 1650, 4GB).

```bash
pip install tensorflow keras keras-preprocessing scikit-learn numpy pandas matplotlib \
            wordcloud nltk gTTS playsound tqdm jupyter
```

### Data setup

Download [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place it under:

```text
./kaggle/Images/         # all .jpg images
./kaggle/captions.txt    # image_id,caption per line
```

### Running the notebook

```bash
jupyter notebook ABICG-Model-Code.ipynb
```

Run cells top to bottom:

1. Load/EDA on images and captions (word cloud, sample caption plots).
2. Preprocess captions and images, tokenize, pad, and build the `tf.data` pipeline.
3. Extract InceptionV3 features for every image once, ahead of training.
4. Define and instantiate the `Encoder`, `Attention_model`, and `Decoder`.
5. Train for the configured number of epochs (checkpoints saved via `tf.train.Checkpoint`), tracking train/test loss.
6. Evaluate: generate captions for held-out images, visualize per-word attention maps over the image, and score outputs with BLEU.

## Training configuration

| Hyperparameter | Value |
| --- | --- |
| Embedding dimension | 256 |
| GRU units | 512 |
| Vocabulary size | 5,000 |
| Batch size | 64 |
| Epochs | 25 |
| Optimizer | Adam, lr = 0.001 |
| Loss | Sparse categorical cross-entropy (masked) |

## Results

| Model | BLEU-based Accuracy | Final Train Loss |
| --- | --- | --- |
| InceptionV3-GRU (no attention) | 79% | 0.8647 |
| **ABICG** (InceptionV3-GRU + Bahdanau attention) | **90%** | **0.0050** |

Example generated captions:

- *"large bird swooping down towards the ground"* (reference: *"a white bird swooping down the ground"*)
- *"two girls hanging upside down on monkey-bars at a park"* (reference: *"two girls are hanging upside down"*)
- *"the people are standing in front of the building"* (reference: *"the people are standing before a building"*)

The notebook also plots per-word attention heatmaps over the source image, showing which region the decoder focused on when generating each token.

## Limitations & future work

- The model is limited by its 5,000-word vocabulary; images requiring out-of-vocabulary concepts fall back to `<UNK>` and can produce trivial captions.
- Trained and evaluated only on Flickr8k — generalization to larger, more diverse datasets (e.g. MS COCO) is untested.
- Planned direction: replace the GRU/Bahdanau encoder-decoder with a Transformer-based architecture (multi-head attention + positional embeddings) to better model word order and long-range dependencies.
