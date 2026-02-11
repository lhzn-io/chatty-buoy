import torch
from moshi.models import loaders

def inspect_mimi():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Mimi on {device}...")
    mimi = loaders.get_mimi(loaders.DEFAULT_MIMI_MODEL)
    mimi.to(device)
    mimi.eval()
    
    print("\n--- Mimi Quantizer Structure ---")
    print(mimi.quantizer)
    
    # helper to print specific attributes
    if hasattr(mimi.quantizer, 'rvq_first'):
        print("\n--- RVQ First ---")
        print(mimi.quantizer.rvq_first)
        # Check codebook sizes
        # The structure is typically SplitResidualVectorQuantizer -> ResidualVectorQuantizer -> VectorQuantizer -> layers
        if hasattr(mimi.quantizer.rvq_first, 'vq'):
             vq = mimi.quantizer.rvq_first.vq
             if hasattr(vq, 'layers'):
                for i, layer in enumerate(vq.layers):
                    print(f"Layer {i} codebook size: {layer._codebook.embedding.size()}")
             else:
                 print("rvq_first.vq has no layers")
        else:
             print("rvq_first has no vq")

    if hasattr(mimi.quantizer, 'rvq_rest'):
        print("\n--- RVQ Rest ---")
        print(mimi.quantizer.rvq_rest)
        if hasattr(mimi.quantizer.rvq_rest, 'vq'):
             vq = mimi.quantizer.rvq_rest.vq
             if hasattr(vq, 'layers'):
                for i, layer in enumerate(vq.layers):
                    print(f"Layer {i} codebook size: {layer._codebook.embedding.size()}")

if __name__ == "__main__":
    inspect_mimi()
