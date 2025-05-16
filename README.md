# Ai-data-bias-detection-and-mitigation
This project demonstrates how to detect and mitigate bias in machine learning models using IBM's AIF360 toolkit alongside scikit-learn and TensorFlow. We use the UCI Adult Census dataset to show how fairness-aware techniques can help reduce discrimination in predictive modeling.
📌 Features
Bias detection based on sensitive attributes (e.g., sex, race)

Fairness metrics (Disparate Impact, Mean Difference)

Bias mitigation using Reweighing (AIF360)

Model training using Logistic Regression (scikit-learn) or deep learning (TensorFlow/PyTorch)

Performance and fairness evaluation before and after mitigation

🛠 Tools & Libraries
Python

IBM AIF360

scikit-learn

TensorFlow (or PyTorch)

pandas, numpy, matplotlib

📁 Project Structure
bash
Copy
Edit
.
|__ project.py             # Main script for bias mitigation
├── requirements.txt       # Dependencies
└── README.md              # Project overview
🚀 How to Run
Install dependencies:


pip install -r requirements.txt
Run the notebook or script:


Edit
python project.py
Output will include:

Model accuracy

Fairness metrics before/after mitigation

🧪 Example Fairness Metrics
Pre-Mitigation Bias (Mean Difference): 0.18

Post-Mitigation Bias (Mean Difference): 0.03

Accuracy: 83.2%

📚 Dataset
https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
