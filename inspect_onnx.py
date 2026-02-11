import onnxruntime as ort
import sys

model_path = "models/silero_vad.onnx"
try:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("Inputs:")
    for i in sess.get_inputs():
        print(f"  Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
    print("\nOutputs:")
    for o in sess.get_outputs():
        print(f"  Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
except Exception as e:
    print(f"Error inspecting model: {e}")
