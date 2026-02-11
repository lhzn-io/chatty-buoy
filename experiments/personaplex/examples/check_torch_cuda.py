import torch

def check_cuda():
    print("--- PyTorch CUDA Check ---")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is NOT available. PyTorch was likely installed without CUDA support.")
        # Attempt to get more diagnostics
        try:
            # This will error if CUDA is not just unavailable but fully unconfigured
            torch.zeros(1).cuda()
        except Exception as e:
            print(f"\nDiagnostic error when trying to use CUDA: {e}")

    print("--- End of Check ---")

if __name__ == "__main__":
    check_cuda()
