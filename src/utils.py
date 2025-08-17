# utils.py
import numpy as np
import joblib


def load_encoders():
    le_ip_src = joblib.load("encoders/le_ip_src.pkl")
    le_ip_dst = joblib.load("encoders/le_ip_dst.pkl")
    le_proto = joblib.load("encoders/le_proto.pkl")
    le_qos = joblib.load("encoders/le_qos.pkl")
    return le_ip_src, le_ip_dst, le_proto, le_qos


# For flat models

def preprocess_flat(data):
    le_ip_src, le_ip_dst, le_proto, _ = load_encoders()  # ✅ FIXED
    src = le_ip_src.transform([data.source_ip])[0]
    dst = le_ip_dst.transform([data.destination_ip])[0]
    proto = le_proto.transform([data.protocol])[0]
    return np.array([[src, dst, proto, data.packet_size, data.inter_arrival_time_ms, data.jitter_ms]])


# For deep models

def preprocess_sequence(data):
    le_ip_src, le_ip_dst, le_proto, _ = load_encoders()
    encoded_seq = []
    for d in data.sequence:
        src = le_ip_src.transform([d.source_ip])[0]
        dst = le_ip_dst.transform([d.destination_ip])[0]
        proto = le_proto.transform([d.protocol])[0]
        row = [src, dst, proto, d.packet_size, d.inter_arrival_time_ms, d.jitter_ms]
        encoded_seq.append(row)
    return np.expand_dims(np.array(encoded_seq), axis=0)

# For IP embedding model

def preprocess_ip_embedding(data):
    le_ip_src, le_ip_dst, le_proto, _ = load_encoders()
    src = le_ip_src.transform([data.source_ip])[0]
    dst = le_ip_dst.transform([data.destination_ip])[0]
    proto = le_proto.transform([data.protocol])[0]
    numeric_features = [data.packet_size, data.inter_arrival_time_ms, data.jitter_ms]
    return {
        "src_input": np.array([[src]]),
        "dst_input": np.array([[dst]]),
        "proto_input": np.array([[proto]]),
        "num_input": np.array([numeric_features])
    }
