# AMD ROCm 기반 AI 개발 및 교육 커리큘럼 가이드

---

## 1. AMD AI 솔루션의 현황 및 경쟁력

AMD의 AI 솔루션은 **독점적 진영(Nvidia CUDA)에 맞서는 가장 강력한 '오픈 생태계 대안'**으로 확고히 자리 잡았습니다. HW 라인업의 빠른 신제품 주기와 SW 생태계의 고도화를 바탕으로 빅테크 기업들의 채용이 지속해서 늘고 있습니다.

### 1) 하드웨어 라인업: 압도적인 메모리 스펙과 1년 단위 신제품 주기
* **MI300X / MI300A:** 192GB HBM3 메모리를 탑재해 출시 직후부터 독점 구도에 균열을 냈습니다. 경쟁 제품(Nvidia H100 80GB) 대비 넉넉한 VRAM 용량 덕분에 대규모 언어 모델(LLM) 추론 및 추론 파이프라인 단축 분야에서 큰 가성비를 보여주었습니다.
* **MI325X:** 288GB HBM3E 메모리를 탑재하여 초거대 모델 추론과 프롬프트 길이가 긴 Long-Context LLM 처리 능력을 끌어올렸습니다.
* **MI350 시리즈 (CDNA 4 아키텍처):** 3nm 공정과 FP4/FP6 데이터 타입을 최초로 지원하며, 기존 MI300 대비 추론 성능을 비약적으로 끌어올렸습니다.
* **MI400 시리즈 (CDNA Next):** 차세대 HBM4 규격 지원 및 스케일아웃 네트워크 강화를 목표로 지속적인 기술 개발이 이어지고 있습니다.

### 2) 오픈소스 소프트웨어 (ROCm) 생태계의 비약적 성장
* **Day-Zero 프레임워크 지원:** PyTorch, TensorFlow 등 주요 AI 프레임워크와 최신 최적화 라이브러리(vLLM, Hugging Face 등)에서 AMD 가속기를 별도 수정 없이 기본 지원(Out-of-the-box)하는 비율이 크게 높아졌습니다.
* **코드 이식성 개선:** CUDA 코드를 ROCm 기반 HIPIFY로 변환하는 도구 수준이 크게 제고되어 기업들이 기존 AI 모델을 AMD 클러스터로 이전하는 진입 장벽이 현저히 낮아졌습니다.

### 3) 클라우드 및 고객사 채용 현황
* **주요 고객사:** Microsoft Azure, Meta, Oracle Cloud, Dell, HPE, Lenovo 등 주요 CSP 및 하드웨어 제조사가 AMD Instinct 기반 서버/클라우드 인스턴스를 적극 제공 중입니다.
* **주요 사용처:** 특히 실시간 서비스 비용 감축이 시급한 **LLM 추론(Inference) 서비스** 부문에서 고용량 메모리 효율성 덕분에 고성능·고효율 옵션으로 인정받고 있습니다.

### 4) 엣지 및 PC 분야 (NPU 통합)
* **Ryzen AI (NPU):** PC용 프로세서에 강력한 AI 가속기(NPU)를 내장하여 Copilot+ PC 및 온디바이스 AI 시장에서 경쟁력을 높였습니다.
* **Embedded & Jetson 대항마:** 임베디드 장치 분야에서는 Xilinx 인수로 확보한 FPGA 기술과 융합한 Kria 시리즈 및 엣지 AI 솔루션을 통해 산업용 AI 시장을 공략하고 있습니다.

---

## 2. AMD ROCm AI 실습 과정 커리큘럼

### 1) 교육 개요 및 목표
* **목표:** CUDA 코드 기반 학습 환경을 AMD ROCm 기반으로 전환하는 원리를 이해하고, PyTorch/HIP 프레임워크를 활용해 최신 LLM 추론 및 AI 파이프라인을 구축한다.
* **대상:** AI/임베디드 엔지니어 및 로보틱스/SW 개발자
* **수업 구성:** 이론 30%, Docker 기반 환경 구축 20%, 실습 프로젝트 50%

### 2) 모듈별 세부 커리큘럼

#### 모듈 1: AMD ROCm 생태계 이해 및 가상화 (8시간)
* AMD CDNA Architecture 및 ROCm 아키텍처 이해
* ROCm vs NVIDIA CUDA 기술 대응 구조 비교 (HIP, rocBLAS 등)
* Docker 환경을 활용한 AMD GPU 패스스루 설정 및 무결성 검증

#### 모듈 2: HIP 프로그래밍 및 코드 마이그레이션 (16시간)
* C++ HIP(Heterogeneous-compute Interface for Portability) 기초
* `HIPIFY` 도구를 이용한 기존 CUDA 소스코드의 ROCm 변환 실습
* Memory Management (Unified Memory / Device Memory) 비교

#### 모듈 3: ROCm 기반 PyTorch 딥러닝 실습 (16시간)
* ROCm PyTorch 컨테이너 환경 내 텐서 컴퓨팅 제어
* Multi-GPU 데이터 병렬화 (`torch.nn.DataParallel` / `DDP`)
* AMD 가속기 최적화 기법 (Mixed Precision, FP16/BF16/FP8 최적화)

#### 모듈 4: LLM 추론 엔진 및 엣지 AI 구축 (16시간)
* ROCm 기반 **vLLM** 및 **Hugging Face** 파이프라인 연동
* LLM 경량화 및 추론 최적화 (vLLM, ComfyUI 연동)
* On-Device/산업용 하드웨어 연동 기초 (Jetson/Kria/Ryzen AI 매핑 개념)

---

## 3. ROCm 개발 환경 구축 및 실습 가이드

본 실습은 **Ubuntu 22.04 LTS / 24.04 LTS** 기준, **Docker** 환경에서 실행하도록 구성되었습니다.

### [실습 1] ROCm 호스트 드라이버 확인 및 Docker 실행

#### 1.1 호스트 GPU 장치 인식 확인
```bash
# GPU 디바이스 및 ROCm 커널 상태 확인
lspci | grep -i amd
ls -l /dev/kfd /dev/dri
```

#### 1.2 PyTorch 전용 ROCm Docker 컨테이너 실행
```bash
# ROCm 최신 PyTorch 이미지를 가져와 컨테이너 실행
docker run -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 16G \
  --name rocm_ai_lab \
  rocm/pytorch:latest
```

---

### [실습 2] ROCm 디바이스 연동 검증 (Python)

컨테이너 내부로 진입한 후, PyTorch에서 AMD GPU가 정상 인식되는지 확인합니다.

```python
# test_rocm.py
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm / CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
    
    # 텐서 연산 테스트 (AMD GPU 상에서 실행)
    x = torch.randn(2000, 2000, device="cuda")
    y = torch.randn(2000, 2000, device="cuda")
    z = torch.matmul(x, y)
    
    print("Matrix Multiplication Succeeded!")
    print(f"Result Tensor Shape: {z.shape}")
else:
    print("ROCm GPU Acceleration not detected.")
```

---

### [실습 3] CUDA 코드를 HIP으로 자동 변환 (HIPIFY)

NVIDIA CUDA 기반 코드를 ROCm 소스코드로 변환하는 명령어 실습입니다.

#### 3.1 기존 CUDA C++ 소스코드 (`vector_add.cu`)
```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
```

#### 3.2 HIPIFY 변환 도구 실행
```bash
# hipify-perl 명령어를 통해 CUDA 코드를 HIP 코드로 자동 변환
hipify-perl vector_add.cu > vector_add.hip.cpp
```

#### 3.3 변환된 HIP 코드 컴파일 및 실행
```bash
# hipcc 컴파일러를 통한 실행 파일 생성
hipcc vector_add.hip.cpp -o vector_add
./vector_add
```

---

### [실습 4] ROCm 기반 LLM 추론 실습 (Hugging Face)

ROCm 파이프라인 상에서 대규모 언어 모델을 로드하여 추론을 진행합니다.

```python
# llm_inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"

# AMD GPU(cuda) 디바이스로 지정하여 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

prompt = "Explain the advantages of AMD ROCm open source AI ecosystem:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 추론 실행
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=== Output ===")
print(response)
```
