from peft import LoraConfig, prepare_model_for_kbit_training  # LoRA 설정 및 8bit 훈련 준비
from transformers import TrainingArguments  # 훈련 관련 설정을 위한 라이브러리
from . import model  # 사전 학습된 모델 로드

# 모델을 훈련 모드로 설정
model.train()

# Gradient Checkpointing 활성화 (메모리 절약, 속도 저하 가능)
model.gradient_checkpointing_enable()

# 8bit/4bit 훈련을 위해 모델 변환 (메모리 절약 및 속도 최적화)
model = prepare_model_for_kbit_training(model)

# 훈련 결과 저장 경로
output_path = "./result/"

# LoRA(저비용 어댑터 훈련) 설정
config = LoraConfig(
    lora_alpha=256,  # LoRA 스케일링 팩터 (높을수록 학습 용량 증가)
    lora_dropout=0.05,  # 드롭아웃 확률 (과적합 방지)
    r=128,  # LoRA 랭크 (저장할 가중치 행렬 차원, 높을수록 성능 향상 가능)
    target_modules=['v_proj', 'up_proj', 'down_proj', 'k_proj', 'o_proj', 'q_proj', 'gate_proj'],  # LoRA 적용할 모듈
    bias="none",  # LoRA에서 편향 데이터 처리 방식 (none: 편향 X, all: 전체 적용, lora_only: LoRA 부분만 적용)
    task_type="CAUSAL_LM"  # 작업 유형 (CAUSAL_LM: GPT와 같은 언어 생성 모델)
)

# 훈련 파라미터 설정
train_params = TrainingArguments(
    output_dir=output_path,  # 모델이 저장될 경로
    num_train_epochs=3,  # 학습 반복 횟수 (Epoch 수)
    per_device_train_batch_size=2,  # 각 GPU당 배치 크기 (메모리 제한 고려)
    gradient_accumulation_steps=1,  # 그라디언트 누적 스텝 수 (실제 배치 크기 확장 효과)
    save_strategy="epoch",  # 매 Epoch마다 체크포인트 저장
    optim="paged_adamw_8bit",  # 8-bit AdamW 최적화 사용 (메모리 절약)
    learning_rate=1e-4,  # 학습률
    logging_steps=100,  # 100 스텝마다 로그 출력
    weight_decay=0.01,  # 가중치 감쇠 (과적합 방지)
    max_grad_norm=0.3,  # Gradient Clipping 값 (최대 그래디언트 값 제한)
    warmup_ratio=0.1,  # 학습률 웜업 비율 (초반 10% 구간에서 점진적으로 학습률 증가)
    fp16=True,  # 16-bit 부동소수점 연산 사용 (속도 및 메모리 절약)
    lr_scheduler_type="cosine",  # 학습률 스케줄러 (Cosine Annealing 적용)
    seed=42  # 랜덤 시드 설정 (재현성 보장)
)
