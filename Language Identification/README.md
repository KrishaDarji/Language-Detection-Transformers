# Audio Processing and Language Identification Projects

This repository contains three Jupyter Notebooks showcasing different approaches to audio processing, automatic speech recognition, and language identification using advanced machine learning and deep learning techniques.

## Notebooks Overview

### 1. Audio Language Classification Using BLSTMs
This notebook explores the use of Bidirectional Long Short-Term Memory (BLSTM) networks for classifying languages based on audio data.

- **Key Features:**
  - Utilizes `librosa` for feature extraction from audio files.
  - Employs `tensorflow.keras` for building and training BLSTM models.
  - Includes data preprocessing, visualization, and model evaluation steps.
  - Hyperparameter tuning is performed with `keras_tuner`.

---

### 2. Whisper-ASR
This notebook demonstrates the usage of OpenAI's Whisper model for Automatic Speech Recognition (ASR).

- **Key Features:**
  - Provides installation instructions for the Whisper ASR library.
  - Implements a pipeline to process audio files for speech-to-text conversion.
  - Leverages the power of Whisper for accurate transcription of speech.

---

### 3. Language Identification Using Wav2Vec2
This notebook focuses on language identification tasks using the Wav2Vec2 model from Hugging Face's Transformers library.

- **Key Features:**
  - Utilizes `torchaudio` and `librosa` for audio processing.
  - Integrates with the Transformers library for feature extraction and modeling.
  - Implements training with the `Trainer` API and fine-tunes the Wav2Vec2 model for language classification.
  - Employs techniques like mixed-precision training with NVIDIA AMP for optimized performance.

---

## Requirements

The notebooks require the following libraries:
- `librosa`, `numpy`, `pandas`, `matplotlib`, `seaborn`
- `tensorflow.keras`, `keras_tuner`
- `torch`, `torchaudio`, `transformers`
- OpenAI Whisper library
- Additional utilities: `datasets`, `tqdm`, `pathlib`, `scikit-learn`

Install the necessary dependencies using `pip` or follow the instructions provided in each notebook.

## How to Run

1. Clone this repository:
   ```bash
   git clone <repository-url>
