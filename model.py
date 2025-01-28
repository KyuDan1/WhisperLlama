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
        
        self.encoder_dim = 1280  # Whisper encoder dimension
        self.decoder_dim = 2048  # Llama hidden dimension
        self.vocab_size = llama_decoder.config.vocab_size
        
        # Bridge network to convert encoder features to decoder dimension
        self.bridge = nn.Sequential(
            nn.Linear(self.encoder_dim, self.decoder_dim),
            nn.GELU(),
            nn.Linear(self.decoder_dim, self.decoder_dim)
        )
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.decoder_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.decoder_dim, self.vocab_size)
        
    def forward(self, input_features, labels=None):
        # Encode audio features
        with torch.no_grad():  # Whisper encoder is frozen
            encoder_outputs = self.encoder(
                input_features=input_features,
                return_dict=True
            )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Transform encoder outputs to decoder dimension
        bridged_features = self.bridge(encoder_hidden_states)
        
        batch_size = input_features.size(0)
        max_length = min(128, labels.size(1) if labels is not None else 128)
        device = input_features.device
        
        # Initialize sequence with BOS token
        current_ids = torch.full((batch_size, 1), self.decoder.config.bos_token_id, device=device)
        
        total_loss = 0
        
        # Autoregressive generation with memory efficient implementation
        for i in range(max_length - 1):
            # Get decoder hidden states for current sequence
            decoder_outputs = self.decoder(
                input_ids=current_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            decoder_hidden_states = decoder_outputs.hidden_states[-1]
            
            # Apply cross-attention only to the last token's hidden state
            last_hidden = decoder_hidden_states[:, -1:, :]
            cross_attn_output, _ = self.cross_attention(
                query=last_hidden,
                key=bridged_features,
                value=bridged_features
            )
            
            # Generate logits for the next token
            combined_features = last_hidden + cross_attn_output
            next_token_logits = self.output_projection(combined_features)
            
            if labels is not None:
                # Calculate loss for this step
                if i < labels.size(1) - 1:
                    loss = F.cross_entropy(
                        next_token_logits.view(-1, self.vocab_size),
                        labels[:, i+1].view(-1),
                        ignore_index=-100
                    )
                    total_loss += loss
            
            # Sample next token
            if labels is not None:
                # Teacher forcing
                next_token = labels[:, i+1:i+2]
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Append next token to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Free up memory
            del decoder_outputs, decoder_hidden_states, cross_attn_output
            torch.cuda.empty_cache()
        
        return type('OutputType', (), {
            'loss': total_loss / (max_length - 1) if labels is not None else None,
            'logits': None,  # We're not storing all logits anymore
            'predictions': current_ids
        })

def create_asr_model(whisper_encoder, llama_decoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Freeze Whisper encoder
    for param in whisper_encoder.parameters():
        param.requires_grad = False
    
    # Move models to device
    whisper_encoder = whisper_encoder.to(device)
    llama_decoder = llama_decoder.to(device)
    
    model = WhisperLlamaASR(whisper_encoder, llama_decoder)
    return model.to(device)

def process_batch(model, input_features):
    device = next(model.parameters()).device
    input_features = input_features.to(device)
    
    outputs = model(input_features=input_features)
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


def generate_text(model, input_features, max_length=100):

    device = next(model.parameters()).device
    input_features = input_features.to(device)
    
    model.eval()
    
    with torch.no_grad():
        # Whisper 인코더와 bridge를 통과
        encoder_outputs = model.encoder(
            input_features=input_features,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        token_logits = model.bridge(encoder_hidden_states)
        initial_tokens = torch.argmax(token_logits, dim=-1)
        
        # Llama로 텍스트 생성
        output_sequences = model.decoder.generate(
            input_ids=initial_tokens,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
    
    return output_sequences


def decode_asr_output(outputs, tokenizer, skip_special_tokens=True):

    # logits에서 가장 높은 확률을 가진 토큰 인덱스 선택
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    # 토큰을 텍스트로 디코딩
    decoded_text = tokenizer.decode(predictions[0], skip_special_tokens=skip_special_tokens)
    
    return decoded_text