---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

````python
# Fashion-MNIST Classifier (Flask Deployment)

A CNN trained on Fashion-MNIST (~94-95% test accuracy) and served through a Flask web app. Upload a clothing image and get the predicted class with the top 3 probabilities.

## Features
- Image upload with live preview
- Top-3 predictions with confidence bars
- Invert option for white-background photos
- Docker and gunicorn ready

## Project structure
```
├── app.py
├── model.keras
├── requirements.txt
└── templates/
    └── index.html
```

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

## Run with Docker
```bash
docker build -t fmnist .
docker run -p 8000:8000 fmnist
```

## Classes
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Tech stack
Python, TensorFlow/Keras, Flask, Pillow, Gunicorn
````
