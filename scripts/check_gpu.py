"""
PyTorch가 GPU를 인식하는지 확인하고, bf16 행렬곱 속도(TFLOPS)를 대략 측정한다.

사용법 (minimind_ko 폴더에서 실행, 학습 중이 아닐 때):
    .venv\\Scripts\\python.exe scripts\\check_gpu.py
"""
import time

import torch

MATRIX_SIZE = 4096
REPEAT = 50


def main():
    print("torch", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
    print(f"VRAM: {props.total_memory / 1024 ** 3:.1f} GB")

    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a @ b  # 첫 실행은 초기화 비용이 섞이므로 측정에서 제외
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(REPEAT):
        a @ b
    torch.cuda.synchronize()
    elapsed = time.time() - start
    # n×n 행렬곱 1회 연산량 = 2·n³ FLOP
    print(f"bf16 matmul: {REPEAT * 2 * MATRIX_SIZE ** 3 / elapsed / 1e12:.1f} TFLOPS")


if __name__ == "__main__":
    main()
