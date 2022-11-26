# Intent Detection for Chatbots Project

## Description

This project is about making an intent detector. That creation of an intent detector can be what you find in the first steps of making a chatbot. The goal here is the following: given a simple textual sentence, classify it within one of the 8 classes that are "in_scope", or recognize it as an "out_of_scope" example. 

## Problem Description

The description of the problem we made was the following: given a textual input, produce a textual label among 9 classes: the 8 in_scope and the 9th out_of_scope one. We made it therefore a multiclass classification problem. On that classification problem, we studied classical machine learning techniques along a deep learning one.

## Algorithm comparison

To run the algorithm comparison, run eval.py the following way with the filename you want to evaluate the models on: 

"""
python eval.py --filename filename
"""

What eval.py does is compare the model's average performances over a dataset. To do so, it uses three metrics that are relevant to a problem:
- Pure accuracy
- A metric that penalizes a model that classifies too many examples as out of scope
- A metric that penalizes a model that classifies too many examples into the "lost luggage" intent 

It does not save the models that are trained. They are purely trained for the evaluation task. Use train.py if you want to save the models.

## Algorithm training 
