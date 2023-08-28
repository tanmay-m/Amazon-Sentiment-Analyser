Fine-Tuned Distilled BERT Model for Sentiment Analysis on Amazon Reviews

This repository contains a fine-tuned Distilled BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis on Amazon reviews. The base model used is the distilbert-base-uncased model, which is a smaller and faster version of the original BERT model, pre-trained on a large corpus of text data.


Model Details

The fine-tuned Distilled BERT model is based on the transformers library by Hugging Face, which provides pre-trained language models that can be fine-tuned on specific tasks. The model architecture used in this repository is the distilbert-base-uncased model, which is a lightweight version of the BERT model with uncased text input. The model is fine-tuned using a binary classification approach, where the goal is to predict whether a given Amazon review is positive or negative based on the text of the review.


Dataset

The model is trained on a dataset of Amazon reviews, which is preprocessed to remove any personally identifiable information (PII) and other irrelevant information. The dataset is split into training, validation, and test sets, with an 80/10/10 split ratio. The training set is used for fine-tuning the model, the validation set is used for hyperparameter tuning, and the test set is used for evaluating the model's performance.


Deployment on Hugging Face

The fine-tuned Distilled BERT model is deployed on Hugging Face's model hub, a platform for hosting and sharing NLP models. The model is available for download and inference through the Hugging Face Transformers library. To use the deployed model, you need to install the transformers library by Hugging Face and load the model using the provided Hugging Face model name or model checkpoint URL.


Here's an example code snippet to load and use the fine-tuned Distilled BERT model for sentiment analysis from Hugging Face:

Copy code
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the fine-tuned model from Hugging Face
model_name = "sohan-ai/sentiment-analysis-model-amazon-reviews"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Tokenize input text
text = "This is a positive review."
inputs = tokenizer(text, return_tensors="pt")

# Make prediction
outputs = model(**inputs)
predicted_label = "positive" if outputs.logits.argmax().item() == 1 else "negative"

print(f"Predicted sentiment: {predicted_label}")


Evaluation Metrics

The performance of the fine-tuned Distilled BERT model can be evaluated using various evaluation metrics, such as accuracy, precision, recall, and F1 score. These metrics can be calculated on the test set of the Amazon reviews dataset to assess the model's accuracy and effectiveness in predicting sentiment.


Conclusion

The fine-tuned Distilled BERT model in this repository, deployed on Hugging Face, provides an accurate and efficient way to perform sentiment analysis on Amazon reviews. It can be used in various applications, such as customer feedback analysis, market research, and sentiment monitoring. Please refer to the Hugging Face Transformers documentation for more details on how to use and fine-tune the Distilled BERT model.
