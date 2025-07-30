
# AI-Driven Security Framework for 5G-enabled IoT Device Management using MEC

# ---------------------- IMPORTS ----------------------
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from sklearn.utils import resample
from matplotlib import pyplot as plt
from flask import Flask, render_template, request
import seaborn as sns
import joblib
import random
import time
import threading

# ---------------------- GLOBAL VARIABLES ----------------------
cloud_log_path = 'cloud_logs.txt'

def log_to_cloud(data):
    with open(cloud_log_path, 'a') as f:
        f.write(data + '\n')

# ---------------------- 1. DATA PREPROCESSING ----------------------
def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop('label', axis=1))
    y = LabelEncoder().fit_transform(df['label'])
    return train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------- 2. INTRUSION DETECTION ----------------------
def train_model(X_train, y_train, model_type="rf"):
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == "svm":
        model = SVC()
    elif model_type == "mlp":
        model = MLPClassifier(max_iter=500)
    elif model_type == "dt":
        model = DecisionTreeClassifier()
    else:
        model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, f'{model_type}_model.pkl')
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------- 3. SECURE OFFLOADING ----------------------
def encrypt_data(data):
    key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
    return key, nonce, ciphertext, tag

def decrypt_data(key, nonce, ciphertext):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)
    return plaintext.decode('utf-8')

# ---------------------- 4. PRIVACY PRESERVATION ----------------------
def add_differential_privacy(data, epsilon=0.5):
    noise = np.random.laplace(loc=0.0, scale=1/epsilon, size=data.shape)
    return data + noise

# ---------------------- 5. FEDERATED LEARNING SIMULATION ----------------------
def federated_learning_sim(X_train, y_train, rounds=5):
    clients = np.array_split(list(zip(X_train, y_train)), 5)
    global_model = LogisticRegression()
    global_model.fit(X_train, y_train)

    for r in range(rounds):
        client_models = []
        print(f"--- Federated Round {r+1} ---")
        for c in clients:
            c = list(c)
            X_c, y_c = zip(*c)
            local_model = LogisticRegression()
            local_model.fit(X_c, y_c)
            client_models.append(local_model.coef_)

        mean_weights = np.mean(client_models, axis=0)
        global_model.coef_ = mean_weights

    return global_model

# ---------------------- 6. REAL-TIME THREAT DETECTION ----------------------
def simulate_realtime_detection(model, sample_pool):
    print("\n--- Real-Time Threat Detection ---")
    for i in range(10):
        sample = random.choice(sample_pool)
        result = model.predict([sample])[0]
        log_to_cloud(f"Real-time check [{i+1}]: {'THREAT' if result == 1 else 'NORMAL'}")
        time.sleep(1)

# ---------------------- 7. VISUALIZATION ----------------------
def visualize_dataset(path):
    df = pd.read_csv(path)
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    print("Correlation heatmap saved.")

# ---------------------- 8. FLASK DASHBOARD ----------------------
app = Flask(__name__)

@app.route('/')
def index():
    with open(cloud_log_path, 'r') as f:
        logs = f.readlines()
    return render_template("index.html", logs=logs)

# ---------------------- MAIN EXECUTION ----------------------
if __name__ == "__main__":
    path_to_dataset = 'TON_IoT_sample.csv'  # Replace with your dataset file
    X_train, X_test, y_train, y_test = load_and_preprocess_data(path_to_dataset)

    model = train_model(X_train, y_train, model_type="rf")
    evaluate_model(model, X_test, y_test)

    X_test_priv = add_differential_privacy(X_test)

    fed_model = federated_learning_sim(X_train, y_train)

    thread = threading.Thread(target=simulate_realtime_detection, args=(model, X_test))
    thread.start()

    visualize_dataset(path_to_dataset)

    payload = 'Live camera data from MEC node'
    key, nonce, ciphertext, tag = encrypt_data(payload)
    decrypted = decrypt_data(key, nonce, ciphertext)
    print("Encrypted Data:", ciphertext)
    print("Decrypted:", decrypted)

    # app.run(debug=True)
    print("\n--- AI-Driven 5G-IoT Security Simulation Complete ---")
