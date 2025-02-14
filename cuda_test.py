import torch
import bitsandbytes as bnb

print("Torch 버전:", torch.__version__)
print("CUDA 사용 가능 여부:", torch.cuda.is_available())
print("CUDA 버전:", torch.version.cuda)
print("GPU 정보:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU 없음")
print("bitsandbytes 버전:", bnb.__version__)

# CUDA 장치의 주요 버전과 부 버전을 가져옵니다.
major_version, minor_version = torch.cuda.get_device_capability()
print(major_version) 
print(minor_version)
