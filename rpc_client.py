import rpyc

MAC_IP = "YOUR_MAC_IP_ADDRESS"  # Replace with your Mac's actual IP

conn = rpyc.connect(MAC_IP, 12345)
with open("temp.wav", "rb") as f:
    data = f.read()

response = conn.root.handle_audio(data)
print("Response:", response)
