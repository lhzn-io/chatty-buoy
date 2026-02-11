import sounddevice as sd
print("Imported sd")
try:
    print(sd.query_devices())
except Exception as e:
    print(f"Error: {e}")
