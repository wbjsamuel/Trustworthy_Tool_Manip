import socket
import pickle
import struct
import torch
import cv2
import numpy as np
# Replace with your actual model import
# from lightning_module import PolicyModule 

def inference_server(host='0.0.0.0', port=9999):
    # --- Model Setup ---
    device = torch.device("cuda")
    # model = PolicyModule.load_from_checkpoint("model.ckpt").to(device).eval()
    print(f"Server listening on {host}:{port} with RTX 5090...")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    while True:
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        data = b""
        payload_size = struct.calcsize("Q") # Unsigned long long (8 bytes)

        try:
            while True:
                # 1. Retrieve message size
                while len(data) < payload_size:
                    packet = conn.recv(4096)
                    if not packet: break
                    data += packet
                
                if not data: break
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                # 2. Retrieve actual payload
                while len(data) < msg_size:
                    data += conn.recv(4096)
                
                msg_data = data[:msg_size]
                data = data[msg_size:]

                # 3. Unpickle and Process
                obs = pickle.loads(msg_data)
                img = cv2.imdecode(obs['img'], cv2.IMREAD_COLOR)
                joints = obs['joints']

                # --- Inference (Replace with your model logic) ---
                # action = model.predict(img, joints)
                action = [0.0] * 7 # Placeholder: 6 joints + 1 gripper
                
                # 4. Send action back
                response = pickle.dumps(action)
                conn.sendall(struct.pack("Q", len(response)) + response)

        except Exception as e:
            print(f"Connection lost: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    inference_server()