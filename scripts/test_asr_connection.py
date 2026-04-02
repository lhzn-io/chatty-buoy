
import riva.client
import sys

def test_asr_connection(uri="localhost:50051"):
    print(f"Connecting to Riva ASR at {uri}...")
    try:
        auth = riva.client.Auth(uri=uri)
        asr_service = riva.client.ASRService(auth)
        config = asr_service.get_config()
        print("Successfully connected to Riva ASR!")
        # print(f"Available models: {config}")
        return True
    except Exception as e:
        print(f"ASR Connection Failed: {e}")
        return False

if __name__ == "__main__":
    if test_asr_connection():
        sys.exit(0)
    else:
        sys.exit(1)
