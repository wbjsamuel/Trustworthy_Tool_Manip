## inference_server.py
import socket
import pickle
import torch
import numpy as np
import argparse
import sys
from deploy_policy import DeployPolicy # Assume your class is in deploy_policy.py

def get_args():
    parser = argparse.ArgumentParser(description="GPU Inference Server for ROS Policy Deployment")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint (.ckpt)")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="IP to bind to (0.0.0.0 listens on all interfaces)")
    parser.add_argument("--port", type=int, default=5000, help="Socket port")
    return parser.parse_args()

def run_server():
    args = get_args()

    # 1. Load Policy
    print(f"[*] Loading policy from: {args.ckpt}")
    try:
        policy = DeployPolicy(args.ckpt)
        print("[+] Model loaded successfully on RTX 5090.")
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        sys.exit(1)

    # 2. Setup Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow immediate reuse of the port after restart
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((args.ip, args.port))
    except PermissionError:
        print(f"[!] Permission denied binding to port {args.port}. Try a higher port or sudo.")
        sys.exit(1)
        
    server_socket.listen(1)
    print(f"[*] Server listening on {args.ip}:{args.port}")

    while True:
        print("[*] Waiting for connection from ROS client (192.168.1.103)...")
        conn, addr = server_socket.accept()
        print(f"[+] Connected to ROS Client at {addr}")
        
        try:
            while True:
                # --- RECEIVE OBSERVATION ---
                # Read the 4-byte header for data length
                data_len_bytes = conn.recv(4)
                if not data_len_bytes: 
                    break
                data_len = int.from_bytes(data_len_bytes, byteorder='big')
                
                # Receive the full pickle payload in chunks
                chunks = []
                bytes_recvd = 0
                while bytes_recvd < data_len:
                    chunk = conn.recv(min(data_len - bytes_recvd, 8192))
                    if not chunk: break
                    chunks.append(chunk)
                    bytes_recvd += len(chunk)
                
                if bytes_recvd < data_len:
                    print("[!] Incomplete data received.")
                    break

                obs_payload = pickle.loads(b"".join(chunks))
                
                # --- INFERENCE ---
                # The policy handles internal history via the deque
                policy.update_obs(obs_payload)
                action = policy.get_action()
                
                # --- SEND ACTION ---
                action_data = pickle.dumps(action)
                conn.sendall(len(action_data).to_bytes(4, byteorder='big'))
                conn.sendall(action_data)
                
        except (ConnectionResetError, BrokenPipeError):
            print("[!] Client disconnected unexpectedly.")
        except Exception as e:
            print(f"[!] Error during inference loop: {e}")
        finally:
            print("[*] Closing connection.")
            conn.close()

if __name__ == "__main__":
    run_server()