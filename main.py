import os
from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
  context = {}
  if request.method == "POST":
    email = request.form["email"]
    types = ["not spam", "spam"]
    input_data = pd.DataFrame([[email]], columns=["text"])
    input_data = vectorizer.transform(input_data["text"])
    predicted_class = clf.predict(input_data)
    context["prediction"] = types[predicted_class[0]]
  return render_template("index.html", context=context)

if __name__ == "__main__":
  df = pd.read_csv("spam_ham_dataset.csv")
  vectorizer = TfidfVectorizer(stop_words="english")
  X = vectorizer.fit_transform(df["text"])
  y = df["label_num"]
  clf = LogisticRegression()
  clf.fit(X, y)
  app.run()
