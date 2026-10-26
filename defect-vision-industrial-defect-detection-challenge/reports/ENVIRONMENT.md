# Environment Audit

Audited locally on 2026-10-16 in a Windows desktop development environment, not a Kaggle notebook.

| Resource | Verified value |
|---|---|
| OS | Windows 10.0.26200 |
| CPU | 12th Gen Intel Core i5-12450H; 8 cores / 12 logical processors |
| RAM | 15.65 GiB |
| NVIDIA GPU | GeForce RTX 3050 Laptop GPU; 4,094 MiB reported VRAM |
| NVIDIA driver | 592.82 |
| Integrated GPU | Intel UHD Graphics |
| Free disk on C: | 39.06 GiB |
| Python | 3.11.9 at `C:\\Users\\saury\\AppData\\Local\\Programs\\Python\\Python311\\python.exe` |
| Torch | 2.12.1+cpu; `torch.cuda.is_available() == False` |
| Available libraries checked | numpy, pandas, Pillow, matplotlib, OpenCV, scikit-learn, PyYAML, torch, torchvision |
| Missing libraries checked | pytest, ImageHash, Kaggle package/CLI |

`nvidia-smi` can see the GPU, but the installed PyTorch build is CPU-only, so no CUDA training can currently run. No dependencies were installed or upgraded during Phase 1. The audit deliberately implements its own dHash rather than adding ImageHash; standard-library `unittest` is used rather than installing pytest.

The original competition data is 76.06 MB according to the user-supplied Kaggle Dataset Description. Generated diagnostics and reports are written under `artifacts/` and `reports/`; raw inputs remain untouched.
