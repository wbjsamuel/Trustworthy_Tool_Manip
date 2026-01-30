## inference_server.py
import socket
import pickle
import torch
import numpy as np
from deploy_policy import DeployPolicy # Assume your class is in deploy_policy.py

def run_server(ip="192.168.1.127", port=5000, ckpt_path="policy.ckpt"):
    # Load Policy
    print(f"Loading policy from {ckpt_path}...")
    policy = DeployPolicy(ckpt_path)
    
    # Setup Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip, port))
    server_socket.listen(1)
    print(f"Server listening on {ip}:{port}")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connected to ROS Client at {addr}")
        try:
            while True:
                # 1. Receive Observation Data
                # First 4 bytes for data length
                data_len_bytes = conn.recv(4)
                if not data_len_bytes: break
                data_len = int.from_bytes(data_len_bytes, byteorder='big')
                
                # Receive the full pickle payload
                chunks = []
                bytes_recvd = 0
                while bytes_recvd < data_len:
                    chunk = conn.recv(min(data_len - bytes_recvd, 4096))
                    if not chunk: break
                    chunks.append(chunk)
                    bytes_recvd += len(chunk)
                
                obs_payload = pickle.loads(b"".join(chunks))
                
                # 2. Inference
                policy.update_obs(obs_payload)
                action = policy.get_action()
                
                # 3. Send Action back
                action_data = pickle.dumps(action)
                conn.sendall(len(action_data).to_bytes(4, byteorder='big'))
                conn.sendall(action_data)
                
        except Exception as e:
            print(f"Connection lost: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    run_server()