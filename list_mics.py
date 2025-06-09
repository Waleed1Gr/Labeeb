import sounddevice as sd

print("🔍 Available audio devices:")
for idx, device in enumerate(sd.query_devices()):
    if device['max_input_channels'] > 0:
        print(f"{idx}: {device['name']}")
