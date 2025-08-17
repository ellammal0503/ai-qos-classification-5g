# train_all_models.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential, Model, save_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Embedding, Flatten, Concatenate, Input
from tensorflow.keras.utils import to_categorical

def prepare_data(path):
    df = pd.read_csv(path)

    # Save encoders
    le_ip_src = LabelEncoder()
    le_ip_dst = LabelEncoder()
    le_proto = LabelEncoder()
    le_qos = LabelEncoder()

    df['source_ip_enc'] = le_ip_src.fit_transform(df['source_ip'])
    df['destination_ip_enc'] = le_ip_dst.fit_transform(df['destination_ip'])
    df['protocol_enc'] = le_proto.fit_transform(df['protocol'])
    df['qos_class_enc'] = le_qos.fit_transform(df['qos_class'])

    os.makedirs("encoders", exist_ok=True)
    joblib.dump(le_ip_src, "encoders/le_ip_src.pkl")
    joblib.dump(le_ip_dst, "encoders/le_ip_dst.pkl")
    joblib.dump(le_proto, "encoders/le_proto.pkl")
    joblib.dump(le_qos, "encoders/le_qos.pkl")

    features = ['source_ip_enc', 'destination_ip_enc', 'protocol_enc',
                'packet_size', 'inter_arrival_time_ms', 'jitter_ms']
    X = df[features]
    y = df['qos_class_enc']
    return X, y, df, le_qos


def train_classical_models(X, y, le_qos):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs("models", exist_ok=True)

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel='rbf', C=1.0, gamma='scale',probability=True, random_state=42))
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    report = {}
    preds_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Check if predict_proba exists (for ROC)
        y_score = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Save trained model
        joblib.dump(model, f"models/{name}.pkl")

        # Store classification report
        report[name] = classification_report(
            y_test, y_pred, target_names=le_qos.classes_, output_dict=True
        )

        # Store predictions for later plotting
        preds_dict[name] = {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_score": y_score
        }

    return report, preds_dict


def train_deep_models(df, le_qos, sequence_length=5):
    # Detect features and target automatically
    features = ["source_ip_enc", "destination_ip_enc", "protocol_enc", 
                "packet_size", "inter_arrival_time_ms", "jitter_ms"]
    target = "qos_class_enc"

    # --- 1️⃣ Create Sequences (sliding window) ---
    X_seq, y_seq = [], []
    df = df.reset_index(drop=True)

    for i in range(len(df) - sequence_length):
        seq = df[features].iloc[i:i+sequence_length].values
        label = df[target].iloc[i+sequence_length - 1]
        X_seq.append(seq)
        y_seq.append(label)

    X_seq = np.array(X_seq)
    y_seq = to_categorical(y_seq, num_classes=len(le_qos.classes_))

    # --- 2️⃣ Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    # --- 3️⃣ Model input/output sizes ---
    input_shape = (sequence_length, len(features))
    num_classes = y_train.shape[1]

    # --- 4️⃣ Model builders ---
    def build_lstm():
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_bilstm():
        model = Sequential()
        model.add(Bidirectional(LSTM(64), input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    models = {
        "lstm": build_lstm(),
        "bilstm": build_bilstm()
    }

    # --- 5️⃣ Train & save reports ---
    report = {}
    preds_dict = {}
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)

        y_score = model.predict(X_test)
        y_pred = np.argmax(y_score, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Save model
        model.save(f"models/{name}.h5")

        report[name] = {
            "accuracy": float(np.mean(y_pred == y_true)),
            "classification_report": classification_report(y_true, y_pred, target_names=le_qos.classes_, output_dict=True)
        }
        preds_dict[name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score
        }

    return report, preds_dict


def train_ip_embedding_model(df):
    # Label encoding
    le_ip_src = LabelEncoder()
    le_ip_dst = LabelEncoder()
    le_proto = LabelEncoder()
    le_qos = LabelEncoder()

    df['src_enc'] = le_ip_src.fit_transform(df['source_ip'])
    df['dst_enc'] = le_ip_dst.fit_transform(df['destination_ip'])
    df['proto_enc'] = le_proto.fit_transform(df['protocol'])
    df['qos_enc'] = le_qos.fit_transform(df['qos_class'])

    # Inputs
    X_ip_src = df['src_enc'].values.reshape(-1, 1)
    X_ip_dst = df['dst_enc'].values.reshape(-1, 1)
    X_proto = df['proto_enc'].values.reshape(-1, 1)
    X_numeric = df[['packet_size', 'inter_arrival_time_ms', 'jitter_ms']].values
    y_int = df['qos_enc'].values
    y = to_categorical(y_int)

    # Split
    X_train_src, X_test_src, \
    X_train_dst, X_test_dst, \
    X_train_proto, X_test_proto, \
    X_train_num, X_test_num, \
    y_train, y_test, y_train_int, y_test_int = train_test_split(
        X_ip_src, X_ip_dst, X_proto, X_numeric, y, y_int, test_size=0.2, random_state=42
    )

    # Model inputs
    src_input = Input(shape=(1,), name='src_input')
    dst_input = Input(shape=(1,), name='dst_input')
    proto_input = Input(shape=(1,), name='proto_input')
    num_input = Input(shape=(3,), name='num_input')

    # Embeddings
    src_emb = Embedding(input_dim=len(le_ip_src.classes_), output_dim=8)(src_input)
    dst_emb = Embedding(input_dim=len(le_ip_dst.classes_), output_dim=8)(dst_input)
    proto_emb = Embedding(input_dim=len(le_proto.classes_), output_dim=4)(proto_input)

    # Merge
    src_flat = Flatten()(src_emb)
    dst_flat = Flatten()(dst_emb)
    proto_flat = Flatten()(proto_emb)
    merged = Concatenate()([src_flat, dst_flat, proto_flat, num_input])

    # Dense layers
    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.3)(x)
    output = Dense(y.shape[1], activation='softmax')(x)

    # Compile
    model = Model(inputs=[src_input, dst_input, proto_input, num_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(
        [X_train_src, X_train_dst, X_train_proto, X_train_num],
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    save_model(model, "models/ip_embed_model.h5")

    # Evaluate
    loss, acc = model.evaluate([X_test_src, X_test_dst, X_test_proto, X_test_num], y_test, verbose=0)

    # Predictions & scores
    y_score = model.predict([X_test_src, X_test_dst, X_test_proto, X_test_num])
    y_pred_int = np.argmax(y_score, axis=1)

    # Decoded reports
    y_true_decoded = le_qos.inverse_transform(y_test_int)
    y_pred_decoded = le_qos.inverse_transform(y_pred_int)
    report = classification_report(y_true_decoded, y_pred_decoded, output_dict=True)

    # Return in consistent format for plotting
    preds_dict = {
        "y_true": y_test_int,
        "y_pred": y_pred_int,
        "y_score": y_score
    }

    return {"accuracy": round(acc, 3), "classification_report": report}, preds_dict, le_qos



def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs("metrics", exist_ok=True)
    plt.savefig(f"metrics/confusion_matrix_{model_name}.png")
    plt.close()

def plot_roc_curve(y_true_bin, y_score, classes, model_name):
    plt.figure(figsize=(6, 4))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        auc_score = roc_auc_score(y_true_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_score:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    os.makedirs("metrics", exist_ok=True)
    plt.savefig(f"metrics/roc_{model_name}.png")
    plt.close()

def train_all(csv_path):
    # Prepare data
    X, y, df, le_qos = prepare_data(csv_path)

    # --- Classical Models ---
    classical_report, classical_preds = train_classical_models(X, y, le_qos)

    # --- Deep Models ---
    deep_scores, deep_preds = train_deep_models(df, le_qos)

    # --- IP Embedding Model ---
    ip_embed_report, ip_embed_preds, le_qos_ip = train_ip_embedding_model(df)

    # === METRICS GENERATION ===
    # Classical
    for model_name, preds in classical_preds.items():
        plot_confusion_matrix(preds["y_true"], preds["y_pred"], le_qos.classes_, model_name)
        if "y_score" in preds:
            y_true_bin = np.eye(len(le_qos.classes_))[preds["y_true"]]
            plot_roc_curve(y_true_bin, preds["y_score"], le_qos.classes_, model_name)

    # Deep Models
    for model_name, preds in deep_preds.items():
        plot_confusion_matrix(preds["y_true"], preds["y_pred"], le_qos.classes_, model_name)
        if "y_score" in preds:
            y_true_bin = np.eye(len(le_qos.classes_))[preds["y_true"]]
            plot_roc_curve(y_true_bin, preds["y_score"], le_qos.classes_, model_name)

    # IP Embedding
    plot_confusion_matrix(ip_embed_preds["y_true"], ip_embed_preds["y_pred"], le_qos_ip.classes_, "ip_embedding")
    y_true_bin = np.eye(len(le_qos_ip.classes_))[ip_embed_preds["y_true"]]
    plot_roc_curve(y_true_bin, ip_embed_preds["y_score"], le_qos_ip.classes_, "ip_embedding")

    return {
        "message": "✅ All models trained, saved, and metrics generated.",
        "classical": classical_report,
        "deep_learning_accuracy": {
            **deep_scores,
            "ip_embedding": ip_embed_report
        },
        "metrics_folder": "metrics/"
    }


# To run standalone
if __name__ == "__main__":
    results = train_all("synthetic_5g_qos_dataset_10000.csv")
    print(results)
