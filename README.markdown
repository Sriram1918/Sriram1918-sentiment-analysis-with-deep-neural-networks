# Sentiment Analysis with Deep Neural Networks

## Overview
This project implements a sentiment analysis system using deep neural networks to classify text as positive or negative. Built with PyTorch, it includes three models—Simple Neural Network, Convolutional Neural Network (CNN), and Long Short-Term Memory (LSTM)—trained on the IMDb Movie Reviews dataset. The system preprocesses text, trains models, and provides a Flask-based web app for real-time sentiment prediction.

## Features
- **Dataset**: IMDb Movie Reviews (50,000 reviews, balanced positive/negative).
- **Models**:
  - Simple Neural Network (bag-of-words).
  - CNN for text classification.
  - LSTM for sequential text processing.
- **Preprocessing**: Tokenization, stopword removal, GloVe embeddings.
- **Deployment**: Flask app for user-input sentiment prediction.
- **Performance**: Up to 85% test accuracy (LSTM).

## Prerequisites
- Python 3.9+
- Git
- Conda or virtualenv
- CUDA-enabled GPU (optional)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sriram1918/Sriram1918-sentiment-analysis-with-deep-neural-networks.git
   cd Sriram1918-sentiment-analysis-with-deep-neural-networks
   ```

2. **Set Up Environment**:
   Using Conda:
   ```bash
   conda env create -f conda_env.yml
   conda activate sentiment_env
   ```
   Or using virtualenv:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   Place the IMDb dataset in `data/raw/`. Download from [Stanford's Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) if not included.

## Dataset
- **Source**: IMDb Movie Reviews (25,000 train, 25,000 test).
- **Format**: Text files, labeled positive (score ≥ 7) or negative (score ≤ 4).
- **Preprocessing**:
  - NLTK tokenization.
  - Stopword and punctuation removal.
  - GloVe embeddings (100D).
  - Padded sequences (200 tokens).

## Model Architecture
1. **Simple Neural Network**:
   - Embedding layer (GloVe).
   - Fully connected layers with ReLU.
   - Sigmoid output for binary classification.
2. **CNN**:
   - Embedding layer.
   - Convolutional layers with varied filter sizes.
   - Max-pooling and dense layer.
3. **LSTM**:
   - Embedding layer.
   - Bidirectional LSTM (128 hidden units).
   - Dense layer with dropout.

## Usage
1. **Train Models**:
   ```bash
   python train.py --model lstm
   ```
   Options: `--model {simple,cnn,lstm}`. Saves to `models/`.

2. **Evaluate Models**:
   ```bash
   python evaluate.py --model lstm --checkpoint models/lstm_model.pth
   ```

3. **Run Flask App**:
   ```bash
   python app.py
   ```
   Visit `http://localhost:5000` for predictions.

4. **Predict Sentiment**:
   ```python
   from predict import predict_sentiment
   text = "This movie was amazing!"
   sentiment = predict_sentiment(text, model="lstm", checkpoint="models/lstm_model.pth")
   print(f"Sentiment: {'Positive' if sentiment > 0.5 else 'Negative'} (Score: {sentiment:.4f})")
   ```

## Project Structure
```
├── data/
│   ├── raw/                # IMDb dataset
│   ├── processed/          # Preprocessed data
├── models/                 # Model checkpoints
├── notebooks/              # Exploratory data analysis
├── app.py                  # Flask app
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── predict.py              # Prediction utility
├── requirements.txt        # Dependencies
├── conda_env.yml           # Conda environment
└── README.md               # This file
```

## Results
- **Simple Neural Network**: ~78% accuracy.
- **CNN**: ~82% accuracy.
- **LSTM**: ~85% accuracy.
- Training time: ~30 minutes (NVIDIA GTX 1080).

## Contributing
1. Fork the repo.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add feature'`).
4. Push (`git push origin feature/your-feature`).
5. Open a pull request.

## References
- Maas, A. L., et al. "Learning Word Vectors for Sentiment Analysis." ACL, 2011.
- Kim, Y. "Convolutional Neural Networks for Sentence Classification." EMNLP, 2014.
- Hochreiter, S., & Schmidhuber, J. "Long Short-Term Memory." Neural Computation, 1997.

## License
MIT License. See `LICENSE` for details.