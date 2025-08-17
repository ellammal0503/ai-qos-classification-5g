import pandas as pd
import numpy as np
import random

# Define QoS classes and protocols
qos_classes = ['URLLC', 'eMBB', 'mMTC']
protocols = ['TCP', 'UDP', 'ICMP']

data = []

for _ in range(100000):  # Generate 1000 samples
    qos = random.choice(qos_classes)
    protocol = random.choice(protocols)
    
    if qos == 'URLLC':
        packet_size = random.randint(200, 1400)
        inter_arrival = np.random.normal(20, 5)
        jitter = np.random.normal(3, 1)
    elif qos == 'eMBB':
        packet_size = random.randint(900, 1500)
        inter_arrival = np.random.normal(50, 10)
        jitter = np.random.normal(6, 2)
    else:  # mMTC
        packet_size = random.randint(100, 800)
        inter_arrival = np.random.normal(100, 30)
        jitter = np.random.normal(5, 1.5)

    data.append({
        'source_ip': f"192.168.{random.randint(0,255)}.{random.randint(1,254)}",
        'destination_ip': f"192.0.2.{random.randint(1,254)}",
        'protocol': protocol,
        'packet_size': packet_size,
        'inter_arrival_time_ms': round(abs(inter_arrival), 2),
        'jitter_ms': round(abs(jitter), 2),
        'qos_class': qos
    })

df = pd.DataFrame(data)
df.to_csv('synthetic_5g_qos_dataset.csv', index=False)
