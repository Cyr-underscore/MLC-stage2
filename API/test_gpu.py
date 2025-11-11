# test_gpu.py
import torch
import sys

print("=" * 50)
print("TEST CONFIGURATION GPU/CUDA")
print("=" * 50)

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Test de mémoire
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM totale: {vram_gb:.1f} GB")
    
    # Test d'allocation
    try:
        test_tensor = torch.zeros(1000, 1000).cuda()
        print("✅ Test d'allocation GPU: RÉUSSI")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Test d'allocation GPU: ÉCHEC - {e}")
else:
    print("❌ CUDA non disponible - Causes possibles:")
    print("   1. Drivers NVIDIA non installés/mis à jour")
    print("   2. CUDA Toolkit non installé")
    print("   3. PyTorch installé sans support CUDA")
    print("   4. Carte GPU non compatible")

print("=" * 50)