import torch
import torch.nn as nn
from ultralytics import YOLO

class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x is [batch, 84, 8400]
        # Transpose to [batch, 8400, 84]
        x = x.transpose(1, 2)
        # Extract boxes [batch, 8400, 4] (cx, cy, w, h)
        cx = x[:, :, 0:1]
        cy = x[:, :, 1:2]
        w = x[:, :, 2:3]
        h = x[:, :, 3:4]
        
        # Convert to x1, y1, x2, y2
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = torch.cat([x1, y1, x2, y2], dim=-1)

        # Extract class scores and compute Sigmoid
        logits = x[:, :, 4:]
        probs = torch.sigmoid(logits)
        
        # Max score and max label
        scores, labels = torch.max(probs, dim=-1, keepdim=True)
        # Concatenate boxes, max score, max label into [batch, 8400, 6]
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)

# To cleanly trace this without the ultralytics Detect layer doing NMS, 
# we rely on the fact that native torch.onnx.export strips it if we just wrap it.
class WrappedModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.ds_out = DeepStreamOutput()
    def forward(self, x):
        # The ultralytics model returns a tuple during trace, the first element is the raw tensor
        y = self.m(x)
        if isinstance(y, tuple) or isinstance(y, list):
            y = y[0]
        return self.ds_out(y)

def main():
    model = YOLO('yolo26n.pt')
    model.model.eval()
    
    # Critical: Enable PyTorch YOLOv11 export mode so the Detect head
    # natively decodes raw anchor distances to cx, cy, w, h and applies sigmoid to classes.
    # If this is False, it outputs meaningless distance gradients.
    for m in model.model.modules():
        if type(m).__name__ == "Detect":
            m.export = True
            m.format = 'onnx'
            
    wrapped = WrappedModel(model.model)
    dummy_input = torch.randn(1, 3, 640, 640)

    print("Exporting safe DeepStream ONNX model...")
    torch.onnx.export(
        wrapped,
        dummy_input,
        "yolo26n.onnx",
        opset_version=17,
        input_names=["input"],
        output_names=["output"]
    )

    print("Simplifying ONNX model...")
    import onnxslim
    import onnx
    model_onnx = onnx.load("yolo26n.onnx")
    model_onnx = onnxslim.slim(model_onnx)
    onnx.save(model_onnx, "yolo26n.onnx")
    print("Export complete!")

if __name__ == "__main__":
    main()
