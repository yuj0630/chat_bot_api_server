# transformer 사용하는 구간입니다.
# 필요한 라이브러리 import
from langchain_community.llms import huggingface_pipeline
import torch
from accelerate import Accelerator

# 추가적인 transformers 라이브러리 import (이미 일부는 위에서 import됨)
from transformers import (
    AutoModelForCausalLM, # Google의 Gemma 2B 모델 (여기서는 사용되지 않음)
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging
)

# 사용할 모델 ID 설정 (Beomi의 한국어-영어 지원 LLaMA-3 모델, 8B 파라미터 크기)
# model_id = "beomi/llama-3-Open-Ko-8B"
model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B" # Google의 Gemma 2B 모델

# 모델 실행 시 사용할 옵션 설정 (현재 CPU에서 실행하도록 설정됨)
model_kwargs = {'device': 'cuda'}

# 토크나이저 로드 (문장을 토큰으로 변환하는 역할)
tokenizer = AutoTokenizer.from_pretrained(model_id)  # 지정한 모델에서 토크나이저 로드
tokenizer.use_default_system_prompt = False

def setup_llm_pipeline():
    
    accelerator = Accelerator()
    
    # CPU 환경에서는 양자화 옵션 제거
    if torch.cuda.is_available():
        # 4비트 양자화 옵션 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=False
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto", 
            offload_folder="./offload",  # 모델을 오프로드할 디스크 경로 설정
            trust_remote_code=True
        )
        
        # 모델을 명시적으로 GPU로 이동
        # model = model.to('cuda')  # 명시적으로 GPU에 모델을 할당
        
    # CPU 환경에서는 양자화 없이 모델 로드  
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto", 
            offload_folder="./offload",  # 모델을 오프로드할 디스크 경로 설정
            trust_remote_code=True
        )
    
    # HuggingFacePipeline 객체 생성
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.5,
        top_p=0.5,
        return_full_text=False,
        max_new_tokens=128
    )
    
    llm = text_generation_pipeline
    
    return llm

# # 모델 설정 변경 (훈련 및 추론 시 캐시 사용 비활성화)
# model.config.use_cache = False  # 이전 결과를 캐시에 저장하지 않도록 설정 (훈련 시 필요할 수 있음)

# # 모델의 병렬 처리 방식 설정 (TPU 또는 다중 GPU를 사용할 경우 필요할 수도 있음)
# model.config.pretraining_tp = 1  # 병렬화 비율 설정 (1이면 기본값)
