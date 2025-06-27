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
import signal

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
        reconnect_delay = 5  # Start with 5 seconds
        max_reconnect_delay = 60  # Max delay of 1 minute
        
        while True:
            try:
                # Use a hardcoded IP (change to your server's IP)
                self.server_ip = "192.168.1.100"  # Replace with actual IP
                
                server_url = f"ws://{self.server_ip}:{self.SERVICE_PORT}"
                print(f"🔄 Connecting to Labeeb server at {server_url}...")
                
                self.websocket = await websockets.connect(server_url)
                print(f"🔗 Connected to Labeeb server!")
                reconnect_delay = 5  # Reset delay after successful connection
    
                # Run all communication tasks concurrently
                await asyncio.gather(
                    self.stream_audio(),
                    self.stream_camera(),
                    self.handle_server_messages(),
                )
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK):
                print(f"Connection closed, reconnecting in {reconnect_delay}s...")
            except Exception as e:
                print(f"❌ Connection error: {e}")
            
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)  # Exponential backoff

    # define a function to detect wake word using snowboy:
    async def detect_wake_word(self, audio_chunk):
        """Detect wake word in audio chunk"""
        # Placeholder for wake word detection logic
        # You can use a library like Snowboy or Porcupine here
        return False  # Always return False for now

    async def stream_audio(self):
        """Continuously stream audio to server with voice activity detection"""
        
        # VAD frame size (must be 10, 20, or 30ms of audio for WebRTC VAD)
        vad_frame_ms = 30
        vad_frame_size = int(self.sample_rate * vad_frame_ms / 1000)
        speaking = False
        silence_frames = 0
        speech_frames = 0
        
        def audio_callback(indata, frames, time, status):
            nonlocal speaking, silence_frames, speech_frames
            
            if self.websocket:
                try:
                    # Convert float32 to int16 if needed
                    if indata.dtype != np.int16:
                        audio_int16 = (indata * 32767).astype(np.int16)
                    else:
                        audio_int16 = indata
                    
                    audio_bytes = audio_int16.tobytes()
                    
                    # Check if this is speech using VAD
                    is_speech = False
                    
                    # Process audio in VAD frame-sized chunks
                    if len(audio_bytes) >= vad_frame_size * 2:
                        for i in range(0, len(audio_bytes) - vad_frame_size * 2, vad_frame_size * 2):
                            frame = audio_bytes[i:i + vad_frame_size * 2]
                            if len(frame) == vad_frame_size * 2:
                                if self.vad.is_speech(frame, self.sample_rate):
                                    is_speech = True
                                    break
                    
                    # Update speech state
                    if is_speech:
                        silence_frames = 0
                        speech_frames += 1
                        
                        # Require minimum speech frames to start speaking state
                        if speech_frames > 3 and not speaking:  # ~90ms of speech
                            speaking = True
                            # Signal that the user has started speaking
                            asyncio.create_task(
                                self.websocket.send(
                                    json.dumps({
                                        "type": "speech_event", 
                                        "event": "start"
                                    })
                                )
                            )
                    else:
                        silence_frames += 1
                        
                        # Long silence after speaking = end of utterance
                        if speaking and silence_frames > 30:  # ~900ms of silence
                            speaking = False
                            speech_frames = 0
                            # Signal that the utterance has ended
                            asyncio.create_task(
                                self.websocket.send(
                                    json.dumps({
                                        "type": "speech_event", 
                                        "event": "end"
                                    })
                                )
                            )
                    
                    # Always send audio data (both speech and silence)
                    asyncio.create_task(
                        self.websocket.send(
                            json.dumps({
                                "type": "audio",
                                "data": base64.b64encode(audio_bytes).decode(),
                                "is_speech": is_speech
                            })
                        )
                    )
                except Exception as e:
                    print(f"Audio processing error: {e}")
        
        with sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
            callback=audio_callback,
            blocksize=vad_frame_size
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

    # * important to revise this code, i'm not focused today...
    
    def start(self):
        loop = asyncio.get_event_loop()
        for signal_name in ('SIGINT', 'SIGTERM'):
            loop.add_signal_handler(
                getattr(signal, signal_name),
                lambda: asyncio.create_task(self.shutdown())
            )
        loop.run_until_complete(self.run())

    async def run(self):
        await self.connect()
        await self.shutdown_event.wait()

    async def shutdown(self):
        print("Shutting down client...")
        if self.camera:
            self.camera.stop()
        if self.websocket:
            await self.websocket.close()
        self.shutdown_event.set()
if __name__ == "__main__":
    client = LabeebClient()
    client.start()
