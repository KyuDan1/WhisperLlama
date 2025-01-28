from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from copy import deepcopy
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
whisper = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

def modify_llama_blocks(model, num_blocks_to_keep=2):
    """
    Llama 모델의 시작과 끝 부분에서 각각 지정된 수의 블록만 남기고 중간 블록들을 제거합니다.
    
    Args:
        model: Llama 모델
        num_blocks_to_keep: 시작과 끝에서 각각 유지할 블록의 수
    
    Returns:
        수정된 모델
    """
    # 모델 복사
    modified_model = deepcopy(model)
    
    # decoder layers 가져오기
    layers = modified_model.model.layers
    
    # 전체 블록 수
    total_blocks = len(layers)
    
    # 유지할 블록의 인덱스
    keep_indices = list(range(num_blocks_to_keep)) + list(range(total_blocks - num_blocks_to_keep, total_blocks))
    
    # 새로운 블록 리스트 생성
    new_layers = torch.nn.ModuleList([layers[i] for i in keep_indices])
    
    # 기존 layers를 새로운 것으로 교체
    modified_model.model.layers = new_layers
    
    return modified_model

def modify_llama_blocks(model, num_blocks_to_keep=2):
    """
    Llama 모델의 시작과 끝 부분에서 각각 지정된 수의 블록만 남기고 중간 블록들을 제거합니다.
    
    Args:
        model: Llama 모델
        num_blocks_to_keep: 시작과 끝에서 각각 유지할 블록의 수
    
    Returns:
        수정된 모델
    """
    # 모델 복사
    modified_model = deepcopy(model)
    
    # decoder layers 가져오기
    layers = modified_model.model.layers
    
    # 전체 블록 수
    total_blocks = len(layers)
    
    # 유지할 블록의 인덱스
    keep_indices = list(range(num_blocks_to_keep)) + list(range(total_blocks - num_blocks_to_keep, total_blocks))
    
    # 새로운 블록 리스트 생성
    new_layers = torch.nn.ModuleList([layers[i] for i in keep_indices])
    
    # 기존 layers를 새로운 것으로 교체
    modified_model.model.layers = new_layers
    
    return modified_model



class WhisperLlamaASR(nn.Module):
    def __init__(self, whisper_encoder, llama_decoder):
        super().__init__()
        self.encoder = whisper_encoder
        self.decoder = llama_decoder
        
        self.encoder_dim = 1280
        self.decoder_dim = 2048
        
        self.bridge = nn.Sequential(
            nn.Linear(self.encoder_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 모든 컴포넌트를 같은 디바이스로 이동
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
    def forward(self, input_features, decoder_input_ids, decoder_attention_mask=None):
        # 입력을 모델과 같은 디바이스로 이동
        device = next(self.parameters()).device
        input_features = input_features.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(device)
            
        # Whisper encoder를 통과
        encoder_outputs = self.encoder(
            input_features=input_features,
            return_dict=True
        )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state
        bridge_hidden_states = self.bridge(encoder_hidden_states)
        
        # Llama decoder에 전달
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            # encoder_hidden_states=bridge_hidden_states,  # cross-attention이 없으므로 제거
            use_cache=False,
            return_dict=True
        )
        
        return decoder_outputs

def create_asr_model(whisper_encoder, llama_decoder):
    # GPU가 있으면 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    whisper_encoder = whisper_encoder.to(device)
    llama_decoder = llama_decoder.to(device)
    
    model = WhisperLlamaASR(whisper_encoder, llama_decoder)
    return model.to(device)

def process_batch(model, input_features, decoder_input_ids, decoder_attention_mask=None):
    # 모델의 디바이스 확인
    device = next(model.parameters()).device
    
    # 입력 데이터를 모델과 같은 디바이스로 이동
    input_features = input_features.to(device)
    decoder_input_ids = decoder_input_ids.to(device)
    if decoder_attention_mask is not None:
        decoder_attention_mask = decoder_attention_mask.to(device)
    
    outputs = model(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask
    )
    return outputs







def decode_asr_output(outputs, tokenizer, skip_special_tokens=True):
    """
    ASR 모델의 출력을 텍스트로 변환합니다.
    
    Args:
        outputs: 모델의 출력 (CausalLMOutputWithPast)
        tokenizer: Llama 토크나이저
        skip_special_tokens: 특수 토큰 스킵 여부
    
    Returns:
        str: 디코딩된 텍스트
    """
    # logits에서 가장 높은 확률을 가진 토큰 인덱스 선택
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    # 토큰을 텍스트로 디코딩
    decoded_text = tokenizer.decode(predictions[0], skip_special_tokens=skip_special_tokens)
    
    return decoded_text


def generate_text(model, text, max_length=100):
    """
    주어진 텍스트에 대해 Llama 모델을 사용하여 텍스트를 생성합니다.
    
    Args:
        model: Llama 모델
        text: 입력 텍스트
        max_length: 생성할 최대 토큰 수
    
    Returns:
        생성된 텍스트
    """
    # 입력 텍스트를 토큰화
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print("input size: ", inputs)
    # GPU가 있다면 GPU로 이동
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    print(inputs)
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 텍스트 생성
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,  # 시작 토큰 ID 추가
        )
    
    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return generated_text

def decode_asr_output(outputs, tokenizer, skip_special_tokens=True):
    """
    ASR 모델의 출력을 텍스트로 변환합니다.
    
    Args:
        outputs: 모델의 출력 (CausalLMOutputWithPast)
        tokenizer: Llama 토크나이저
        skip_special_tokens: 특수 토큰 스킵 여부
    
    Returns:
        str: 디코딩된 텍스트
    """
    # logits에서 가장 높은 확률을 가진 토큰 인덱스 선택
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    # 토큰을 텍스트로 디코딩
    decoded_text = tokenizer.decode(predictions[0], skip_special_tokens=skip_special_tokens)
    
    return decoded_text

"""modified_llama = modify_llama_blocks(llama, num_blocks_to_keep=2)
tokenizer.padding_side = "left"  # 왼쪽에 패딩 추가

# 모델 사용 전 패딩 토큰 설정
modified_llama.config.pad_token_id = tokenizer.pad_token_id
modified_llama.resize_token_embeddings(len(tokenizer))


whisper_encoder = whisper.model.encoder
asr_model = create_asr_model(whisper_encoder, modified_llama)

# 예시 입력 데이터 생성 (CPU에서 생성)
input_features = torch.randn(1, 80, 3000)
decoder_input_ids = torch.randint(0, 32000, (1, 100))

# 모델 실행 (process_batch 내에서 GPU로 이동)
outputs = process_batch(asr_model, input_features, decoder_input_ids)


tokenizer.pad_token = tokenizer.eos_token

# 직접 디코딩
text = decode_asr_output(outputs, tokenizer)
print("디코딩된 텍스트:", text)
"""