import rpyc

class MyService(rpyc.Service):
    def exposed_handle_audio(self, audio_bytes):
        print("Received audio length:", len(audio_bytes))
        return "Processed on Mac!"

if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    server = ThreadedServer(MyService, port=12345)
    print("RPC server started on port 12345...")
    server.start()
