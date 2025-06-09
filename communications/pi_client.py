import asyncio
import websockets
import json
import socket
import time
import sounddevice as sd
import numpy as np
import base64
from picamera2 import Picamera2
import io

class LabeebClient:
    DISCOVERY_PORT = 5678
    SERVICE_PORT = 6789

    def __init__(self):
        self.server_ip = None
        self.websocket = None
        self.setup_audio()
        self.setup_camera()

    def setup_audio(self):
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = 'int16'

    def setup_camera(self):
        self.camera = Picamera2()
        self.camera.start()

    async def discover_server(self):
        """Find Labeeb server on network"""
        discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        discovery_socket.settimeout(2)

        for _ in range(5):  # Try 5 times
            try:
                # Broadcast discovery message
                discovery_socket.sendto(b'LABEEB_DISCOVER', ('<broadcast>', self.DISCOVERY_PORT))
                data, addr = discovery_socket.recvfrom(1024)
                
                if data == b'LABEEB_SERVER':
                    self.server_ip = addr[0]
                    print(f"✨ Found Labeeb server at {self.server_ip}")
                    return True
            except socket.timeout:
                print("🔄 Searching for server...")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ Discovery error: {e}")
                await asyncio.sleep(1)
        
        return False

    async def connect(self):
        """Connect to server with auto-discovery"""
        while True:
            try:
                if not self.server_ip:
                    if not await self.discover_server():
                        print("❌ Could not find server, retrying...")
                        continue

                server_url = f"ws://{self.server_ip}:{self.SERVICE_PORT}"
                self.websocket = await websockets.connect(server_url)
                print(f"🔗 Connected to Labeeb server at {server_url}")
                
                await asyncio.gather(
                    self.stream_audio(),
                    self.stream_camera(),
                    self.handle_server_messages()
                )
            except Exception as e:
                print(f"❌ Connection error: {e}")
                self.server_ip = None  # Reset server IP to trigger rediscovery
                await asyncio.sleep(5)

    async def stream_audio(self):
        """Continuously stream audio to server"""
        def audio_callback(indata, frames, time, status):
            if self.websocket:
                audio_bytes = indata.tobytes()
                asyncio.create_task(self.websocket.send(json.dumps({
                    "type": "audio",
                    "data": base64.b64encode(audio_bytes).decode()
                })))

        with sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
            callback=audio_callback
        ):
            while True:
                await asyncio.sleep(0.1)

    async def stream_camera(self):
        """Stream camera when motion detected"""
        while True:
            frame = self.camera.capture_array()
            # Basic motion detection
            if self.detect_motion(frame):
                img_bytes = self.frame_to_bytes(frame)
                await self.websocket.send(json.dumps({
                    "type": "camera",
                    "data": base64.b64encode(img_bytes).decode()
                }))
            await asyncio.sleep(0.1)

    async def handle_server_messages(self):
        """Handle responses from server"""
        while True:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "response":
                    # Play audio response
                    self.play_audio(base64.b64decode(data["audio"]))
                elif data["type"] == "detection":
                    # Handle detection results
                    print(f"Detection: {data['result']}")
                    
            except websockets.exceptions.ConnectionClosed:
                print("Connection lost, reconnecting...")
                await self.connect()

    def start(self):
        """Start the client"""
        asyncio.get_event_loop().run_until_complete(self.connect())

if __name__ == "__main__":
    client = LabeebClient()
    client.start()