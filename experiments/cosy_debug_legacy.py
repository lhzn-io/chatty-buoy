
try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    print("Import 1 Success")
except ImportError:
    print("Import 1 Failed")

try:
    from cosyvoice.api.cosyvoice import CosyVoice
    print("Import 2 Success")
except ImportError:
    print("Import 2 Failed")

try:
    import cosyvoice
    print(f"CosyVoice Dir: {cosyvoice.__file__}")
except ImportError:
    print("CosyVoice Pkg Missing")
