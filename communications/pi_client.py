import asyncio
import websockets
import json
import time
import sounddevice as sd
import simpleaudio as sa
import numpy as np
import base64
from picamera2 import Picamera2
import cv2
import webrtcvad
import openwakeword.model as Model

class LabeebClient:
    DISCOVERY_PORT = 5678
    SERVICE_PORT = 6789
    vad = webrtcvad.Vad(1)

    def __init__(self):
        # * add the server IP here:
        self.server_ip = None
        self.websocket = None
        self.setup_audio()
        self.setup_camera()

    def setup_audio(self):
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = "int16"

    def setup_camera(self):
        self.camera = Picamera2()
        self.camera.start()
#! useless unless deemed necessary later for auto finding the server.
    # async def discover_server(self):
    #     """Find Labeeb server on network"""
    #     discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    #     discovery_socket.settimeout(2)

    #     for _ in range(5):  # Try 5 times
    #         try:
    #             # Broadcast discovery message
    #             discovery_socket.sendto(
    #                 b"LABEEB_DISCOVER", ("<broadcast>", self.DISCOVERY_PORT)
    #             )
    #             data, addr = discovery_socket.recvfrom(1024)

    #             if data == b"LABEEB_SERVER":
    #                 self.server_ip = addr[0]
    #                 print(f"✨ Found Labeeb server at {self.server_ip}")
    #                 return True
    #         except socket.timeout:
    #             print("🔄 Searching for server...")
    #             await asyncio.sleep(1)
    #         except Exception as e:
    #             print(f"❌ Discovery error: {e}")
    #             await asyncio.sleep(1)

    #     return False

    async def connect(self):
        """Connect to server with fixed IP"""
        while True:
            try:
                # Use a hardcoded IP (change to your server's IP)
                self.server_ip = "192.168.1.100"  # Replace with your actual server IP
                
                server_url = f"ws://{self.server_ip}:{self.SERVICE_PORT}"
                print(f"🔄 Connecting to Labeeb server at {server_url}...")
                
                self.websocket = await websockets.connect(server_url)
                print(f"🔗 Connected to Labeeb server!")
    
                # Run all communication tasks concurrently
                await asyncio.gather(
                    self.stream_audio(),
                    self.stream_camera(),
                    self.handle_server_messages(),
                )
            except Exception as e:
                print(f"❌ Connection error: {e}")
                print(f"⏳ Retrying in 5 seconds...")
                await asyncio.sleep(5)  # Wait before retry

    # define a function to detect wake word using snowboy:
    async def detect_wake_word(self, audio_chunk):
        """Detect wake word in audio chunk"""
        # Placeholder for wake word detection logic
        # You can use a library like Snowboy or Porcupine here
        return False  # Always return False for now

    async def stream_audio(self):
        """Continuously stream audio to server"""

        def audio_callback(indata, frames, time, status):
            if self.websocket:
                audio_bytes = indata.tobytes()
                asyncio.create_task(
                    self.websocket.send(
                        json.dumps(
                            {
                                "type": "audio",
                                "data": base64.b64encode(audio_bytes).decode(),
                            }
                        )
                    )
                )

        with sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
            callback=audio_callback,
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
                await self.websocket.send(
                    json.dumps(
                        {"type": "camera", "data": base64.b64encode(img_bytes).decode()}
                    )
                )
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

    def play_audio(self, audio_bytes):
        """Play raw PCM audio bytes (16-bit, mono, 16kHz)"""
        try:
            play_obj = sa.play_buffer(audio_bytes, 1, 2, self.sample_rate)
            play_obj.wait_done()
        except Exception as e:
            print(f"❌ Audio playback error: {e}")

    def detect_motion(self, frame):
        """Basic motion detection (placeholder, always True)"""
        # You can implement real motion detection here
        return True

    def frame_to_bytes(self, frame):
        """Convert camera frame (numpy array) to JPEG bytes"""
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()

    def start(self):
        """Start the client"""
        asyncio.get_event_loop().run_until_complete(self.connect())


if __name__ == "__main__":
    client = LabeebClient()
    client.start()
