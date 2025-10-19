# 🧩 AI Development Project — Classical ML, Deep Learning & NLP

## 📘 Overview

This project demonstrates applied AI development across three major domains:

1. **Classical Machine Learning** – *Iris Species Prediction using Scikit-learn*  
2. **Deep Learning** – *Handwritten Digit Classification with TensorFlow CNN*  
3. **Natural Language Processing** – *Product Review Analysis using spaCy*  

It also includes sections on **ethical considerations** and **debugging TensorFlow models**, showing both technical and responsible AI development skills.

---

## ⚙️ Project Structure

```bash
AITools/
│
├── part2.ipynb      # Classical ML task, # CNN Deep Learning task, # NLP with spaCy
├── part1.pdf       
├── part3.pdf                               
└── README.md                           # Project documentation
🧠 Task 1: Classical ML with Scikit-learn

Dataset: Iris Species Dataset

Goal: Train a Decision Tree Classifier to predict iris species.

🔹 Steps Performed

Loaded and explored the dataset using pandas and scikit-learn.

Checked and handled missing values.

Encoded labels using LabelEncoder.

Split data into training and test sets.

Trained a Decision Tree Classifier.

Evaluated using accuracy, precision, and recall metrics.

✅ Results

Accuracy typically >95% on test data.

High precision and recall across all three species.

🤖 Task 2: Deep Learning with TensorFlow (MNIST CNN)

Dataset: MNIST Handwritten Digits

Goal: Build and train a CNN model achieving >95% test accuracy.

🧱 Model Architecture
Layer Type	Details
Input	(28 × 28 × 1) grayscale images
Conv2D + ReLU	32 filters, 3×3 kernel
MaxPooling2D	2×2 pooling window
Conv2D + ReLU	64 filters, 3×3 kernel
MaxPooling2D	2×2 pooling window
Flatten	Convert 2D to 1D
Dense + ReLU	128 neurons
Output (Softmax)	10 neurons (digits 0–9)
⚙️ Configuration

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Epochs: 5

Batch Size: 64

📊 Results

✅ Test Accuracy: 97%+

Visualized predictions for 5 random test samples with predicted and true labels.

💬 Task 3: NLP with spaCy (Amazon Reviews)

Dataset: Sample Amazon Product Reviews
Goal:

Perform Named Entity Recognition (NER) to extract product names and brands.

Conduct Rule-Based Sentiment Analysis (Positive / Negative / Neutral).

🧩 Implementation Highlights

Used spaCy small English model (en_core_web_sm).

Created a PhraseMatcher for product/brand detection.

Built a custom sentiment analyzer using positive/negative keyword lists.

🧾 Example Output
Review: I love my new Apple iPhone 14, the camera quality is amazing!
Extracted Entities: ['Apple', 'iPhone 14']
Sentiment: Positive
⚖️ Ethical Considerations
🔍 Potential Model Biases
Model	Potential Bias	Impact
MNIST CNN	Uneven representation of handwriting styles	Poor accuracy for certain handwriting groups
Amazon Reviews Model	Language and cultural bias in sentiment interpretation	Misclassification of reviews written in diverse dialects
🧰 Mitigation Tools

TensorFlow Fairness Indicators: Evaluate fairness metrics across user groups.

spaCy Rule-Based Systems: Customize text preprocessing rules to neutralize biased terms.

🧩 Troubleshooting Challenge
🐞 Issue

TensorFlow script errors:

Dimension mismatches (missing Flatten() layer).

Wrong activation function (sigmoid instead of softmax).

Incorrect loss (binary_crossentropy for multi-class).

🧠 Fix Implemented

Added Flatten() layer for input reshaping.

Replaced activation → softmax.

Changed loss → sparse_categorical_crossentropy.

✅ Result: Model now trains and evaluates successfully on MNIST data.

🧰 Dependencies
tensorflow==2.16.1
scikit-learn==1.5.0
pandas==2.2.2
matplotlib==3.9.0
spacy==3.7.5

🧭 Learning Outcomes

Implemented Scikit-learn for classical machine learning workflows.

Built and optimized CNN with TensorFlow for image classification.

Utilized spaCy for entity extraction and sentiment analysis.

Applied AI fairness and debugging principles.

🏁 Conclusion

This project demonstrates the end-to-end AI development pipeline — covering data preprocessing, model training, evaluation, and bias mitigation.
It highlights practical technical proficiency, ethical awareness, and reproducibility in modern AI systems.
