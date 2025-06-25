import asyncio
import websockets
import json
import socket
import base64
import time
from pathlib import Path

# Import Labeeb's core functionality
from tasks.task_manager import (
    load_tasks,
    add_task,
    search_tasks,
    delete_task,
    classify_input,
    chat_response,
)
from models.model import whisper_model, sentence_model
from audio.speak import speak, stop_current_speech, generate_speech_bytes
from audio.listen import wait_for_wake_word
from vision.camera import phone_person_detector
from utils.config import client, TEMP_DIR, session, SessionState


class LabeebServer:
    DISCOVERY_PORT = 5678
    SERVICE_PORT = 6789

    def __init__(self, host="0.0.0.0", port=6789):
        self.host = host
        self.port = port
        print("🔰 Loading Labeeb components...")
        load_tasks()  # Load existing tasks on startup
        print("⚙️ Initializing Whisper...")
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.session = SessionState()  # Use the session state management
        self.clients = set()
    
    # * this is from chatGPT, and might replace the process_audio function.
    async def handler(self, websocket, path):
        self.clients.add(websocket)
        print("🔗 Client connected.")
        
        try:
            audio_buffer = deque(maxlen=60)  # 60 chunks ≈ ~1.8 sec
            utterance_buffer = bytearray()
            speech_started = False

            async for message in websocket:
                data = json.loads(message)

                # Audio chunk
                if data["type"] == "audio":
                    chunk = base64.b64decode(data["data"])
                    is_speech = data.get("is_speech", False)
                    timestamp = time.time()

                    audio_buffer.append((timestamp, chunk))
                    if speech_started:
                        utterance_buffer.extend(chunk)

                # Speech events
                elif data["type"] == "speech_event":
                    event = data["event"]

                    if event == "start":
                        print("🟢 Speech started")

                        # Capture recent buffer (rollback)
                        rollback_ms = 500
                        now = time.time()
                        pre_speech = b''.join(
                            chunk for t, chunk in audio_buffer
                            if (now - t) * 1000 <= rollback_ms
                        )

                        utterance_buffer = bytearray(pre_speech)
                        speech_started = True

                        # Stop current speech if any
                        stop_current_speech()

                    elif event == "end":
                        print("🔴 Speech ended")
                        speech_started = False

                        if not utterance_buffer:
                            print("⚠️ Empty utterance, skipping")
                            continue

                        # Process the audio using Labeeb core
                        response = await self.process_audio(bytes(utterance_buffer))

                        if response["type"] == "response":
                            audio_bytes = generate_speech_bytes(response["text"])
                            await websocket.send(json.dumps({
                                "type": "tts",
                                "audio": base64.b64encode(audio_bytes).decode(),
                                "text": response["text"]
                            }))
                        elif response["type"] == "wake":
                            if response.get("session_active"):
                                audio_bytes = generate_speech_bytes(response["text"])
                                await websocket.send(json.dumps({
                                    "type": "tts",
                                    "audio": base64.b64encode(audio_bytes).decode(),
                                    "text": response["text"]
                                }))
                        else:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "text": response["text"]
                            }))

                        # Clear buffer after handling
                        utterance_buffer.clear()
                        audio_buffer.clear()

        except Exception as e:
            print(f"❌ Handler error: {e}")
        finally:
            self.clients.remove(websocket)
            print("❌ Client disconnected")
        
    async def process_audio(self, audio_bytes):
        temp_path = self.temp_dir / f"input_{int(time.time())}.wav"
        try:
            # Save incoming audio
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)

            # Check for wake word if session not active
            if not self.session.active:
                result = whisper_model.transcribe(
                    str(temp_path),
                    language="ar",
                    initial_prompt="لبيب هو اسم الروبوت. الكلمات المتوقعة: لبيب",
                )
                if "لبيب" in result["text"].lower():
                    self.session.active = True
                    return {
                        "type": "wake",
                        "text": "نعم، كيف اقدر اخدمك؟",
                        "session_active": True,
                    }
                return {"type": "wake", "session_active": False}

            # Process normal input
            result = whisper_model.transcribe(
                str(temp_path),
                language="ar",
                initial_prompt="توقع كلام باللهجة السعودية",
            )
            text = result["text"].strip()
            print("📄 النص:", text)

            # Get intent
            intent = classify_input(text)

            # Generate response based on intent
            if intent == "تسجيل":
                add_task(text)
                response_text = "تم تسجيل المهمة"
            elif intent == "تذكير":
                related = search_tasks(text)
                response_text = chat_response(text, related)
            else:
                response_text = chat_response(text, [])

            # Check for conversation end
            if "<close_conversation>" in response_text:
                self.session.active = False
                return {
                    "type": "response",
                    "text": response_text.replace("<close_conversation>", ""),
                    "session_active": False,
                }

            return {"type": "response", "text": response_text, "session_active": True}

        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return {
                "type": "error",
                "text": "حصل خطأ في معالجة الصوت",
                "session_active": self.session.active,
            }
        finally:
            if temp_path.exists():
                temp_path.unlink()

    async def process_image(self, image_bytes):
        temp_path = self.temp_dir / f"image_{int(time.time())}.jpg"
        try:
            with open(temp_path, "wb") as f:
                f.write(image_bytes)
            result = phone_person_detector(str(temp_path))
            return {"type": "detection", "result": result}
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return {"type": "error", "result": "error"}
        finally:
            if temp_path.exists():
                temp_path.unlink()
    # ! this might be deprecated in favor of the new handler method.
    # async def handler(self, websocket, path):
    #     self.clients.add(websocket)
    #     try:
    #         async for message in websocket:
    #             data = json.loads(message)
    #             if data["type"] == "audio":
    #                 audio_bytes = base64.b64decode(data["data"])
    #                 # Process audio: ASR, intent, TTS, etc.
    #                 response_audio = await self.process_audio(audio_bytes)
    #                 await websocket.send(
    #                     json.dumps(
    #                         {
    #                             "type": "response",
    #                             "audio": base64.b64encode(response_audio).decode(),
    #                         }
    #                     )
    #                 )
    #             elif data["type"] == "camera":
    #                 img_bytes = base64.b64decode(data["data"])
    #                 # Process image: detection, etc.
    #                 detection_result = await self.process_image(img_bytes)
    #                 await websocket.send(
    #                     json.dumps({"type": "detection", "result": detection_result})
    #                 )
    #     finally:
    #         self.clients.remove(websocket)

    async def discovery_service(self):
        """Listen for discovery broadcasts"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.DISCOVERY_PORT))
        sock.settimeout(1)
        print("👂 Listening for discovery messages...")
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                if data == b"LABEEB_DISCOVER":
                    print(f"📡 Discovery request from {addr}")
                    sock.sendto(b"LABEEB_SERVER", addr)
            except socket.timeout:
                await asyncio.sleep(0.1)

    def start(self):
        """Start both discovery and WebSocket services"""
        loop = asyncio.get_event_loop()
        server = websockets.serve(
            self.handler, self.host, self.port, ping_interval=None
        )
        print(f"🚀 Server running on ws://{self.host}:{self.port}")
        loop.create_task(self.discovery_service())
        loop.run_until_complete(server)
        loop.run_forever()


if __name__ == "__main__":
    server = LabeebServer()
    server.start()
