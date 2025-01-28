import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForCausalLM
import soundfile as sf
from model import create_asr_model, modify_llama_blocks, decode_asr_output
import gc
import librosa
import numpy as np
def load_trained_model(model_path):
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.5)
        
        print("Loading Whisper encoder...")
        whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v2",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"  # 자동으로 메모리 관리
        )
        whisper_encoder = whisper.get_encoder()
        
        print("Loading Llama...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            use_fast=True
        )
        
        # 토크나이저 설정
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Llama 모델 설정
        llama = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"  # 자동으로 메모리 관리
        )
        llama.config.pad_token_id = tokenizer.pad_token_id
        llama.resize_token_embeddings(len(tokenizer))
        
        modified_llama = modify_llama_blocks(llama, num_blocks_to_keep=2)
        del llama
        gc.collect()
        
        print("Creating model...")
        model = create_asr_model(whisper_encoder, modified_llama)
        model = model.half()
        
        print("Loading weights...")
        state_dict = torch.load(model_path, map_location='cpu')
        
        # 디버깅 정보 출력
        print(f"\nModel vocab size: {model.decoder.model.embed_tokens.weight.shape[0]}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"BOS token id: {tokenizer.bos_token_id}")
        print(f"EOS token id: {tokenizer.eos_token_id}")
        print(f"PAD token id: {tokenizer.pad_token_id}")
        
        missing, unexpected = model.load_state_dict(
            {k: v.half() for k, v in state_dict.items()}, 
            strict=False
        )
        
        print(f"\nMissing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")

        model.eval()
        
        return model, processor, tokenizer
        
    except Exception as e:
        print(f"Error during model loading: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        raise

def process_audio(audio_path, processor):
    try:
        print(f"Loading audio from {audio_path}...")
        # librosa를 사용하여 자동 리샘플링
        audio, orig_sr = librosa.load(audio_path)
        
        # 16kHz로 리샘플링
        if orig_sr != 16000:
            print(f"Resampling from {orig_sr}Hz to 16000Hz")
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
        
        # 오디오 정규화
        audio = audio / np.abs(audio).max()
        
        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.half()
        
        return input_features
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        raise

def run_inference(model, input_features, tokenizer, max_length=200):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                if torch.cuda.is_available():
                    model = model.to(device)
                    input_features = input_features.to(device)
                
                print("\nInput features shape:", input_features.shape)
                
                # 시작 토큰 설정
                start_token = tokenizer.bos_token_id
                print(f"Using start token: {start_token} ({tokenizer.decode([start_token])})")
                
                decoder_input_ids = torch.tensor([[start_token]], 
                                               device=device,
                                               dtype=torch.long)
                
                # Greedy decoding
                max_length = 100
                generated_ids = []
                
                for _ in range(max_length):
                    outputs = model(
                        input_features=input_features,
                        decoder_input_ids=decoder_input_ids
                    )
                    
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                    
                    # Top 5 토큰 출력
                    top_tokens = torch.topk(next_token_logits[0], 5)
                    print("\nTop 5 tokens for position", len(generated_ids))
                    for token_id, logit in zip(top_tokens.indices, top_tokens.values):
                        token = tokenizer.decode([token_id])
                        prob = torch.softmax(top_tokens.values, dim=-1)[0].item()
                        print(f"Token: {token}, Probability: {prob:.4f}")
                    
                    generated_ids.append(next_token_id)
                    
                    if next_token_id == tokenizer.eos_token_id:
                        break
                        
                    decoder_input_ids = torch.cat([
                        decoder_input_ids, 
                        torch.tensor([[next_token_id]], device=device)
                    ], dim=-1)
                
                # 전체 시퀀스 디코딩
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                if torch.cuda.is_available():
                    model = model.cpu()
                    torch.cuda.empty_cache()
                
                return text
                
    except Exception as e:
        print(f"Error during inference: {e}")
        torch.cuda.empty_cache()
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    try:
        model_path = "../models/best_model_epoch_0.pt"
        audio_path = "test.wav"
        
        print("Loading model...")
        model, processor, tokenizer = load_trained_model(model_path)
        
        print("Processing audio...")
        input_features = process_audio(audio_path, processor)
        
        print("Running inference...")
        text = run_inference(model, input_features, tokenizer)
        
        print("\nTranscription:")
        print("-" * 50)
        print(text)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()