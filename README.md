# Project Sentimental

A Python project that trains and runs a sentiment analysis model to classify text into emotional categories such as positive, negative, and neutral.

## Overview

This project uses Hugging Face Transformers with PyTorch to load a pre-trained model and tokenizer, process input text, and predict sentiment. It demonstrates a full Python-based sentiment prediction workflow using modern NLP tools.

## Tech Stack

- Python
- Hugging Face Transformers
- PyTorch
- datasets & evaluate libraries
- CLI interface for inference

## Features

- Train and load a sentiment classifier model
- Input text via CLI (`src/cli.py`)
- Predict emotion/sentiment labels
- Works without web deployment

## Project Structure

```bash
project_sentimental/
├── data/ # Datasets or training data
├── src/ # Source code
│ ├── cli.py # Main CLI entrypoint
│ └── ... # Model loading and inference
├── .gitignore
├── README.md
├── requirements.txt
```

## Installation

- python -m venv .venv
- .venv\Scripts\activate # Windows PowerShell
- pip install -r requirements.txt

## Usage

- python src/train.py
- python src/cli.py

## Author

H2SO4-1191 – Software Engineer
