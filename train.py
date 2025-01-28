import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import WhisperProcessor
import torch.optim as optim
from tqdm import tqdm
import wandb
import numpy as np
from transformers import AutoTokenizer
import torch.nn.functional as F
from copy import deepcopy
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch.nn as nn
from transformers import AutoModelForCausalLM

from model import *

processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
whisper = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

class LibriSpeechDataset(Dataset):
    def __init__(self, split="train.100", processor=None):
        print("lrbirispeech class load dataset...")
        self.dataset = load_dataset("fixie-ai/librispeech_asr", data_dir="clean", split=split)
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 오디오 데이터 로드
        audio = self.dataset[idx]["audio"]
        text = self.dataset[idx]["text"]
        
        # Whisper 프로세서로 오디오 특성 추출
        input_features = self.processor(
            audio["array"],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.squeeze(0)
        
        # 텍스트 토큰화
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_features": input_features,
            "decoder_input_ids": labels["input_ids"].squeeze(0),
            "decoder_attention_mask": labels["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0)
        }

def collate_fn(batch):
    # 배치 내의 최대 길이에 맞춰 패딩
    max_audio_len = max(x["input_features"].shape[1] for x in batch)
    max_text_len = max(x["decoder_input_ids"].shape[0] for x in batch)
    
    input_features = []
    decoder_input_ids = []
    decoder_attention_mask = []
    labels = []
    
    for item in batch:
        # 오디오 특성 패딩
        padded_audio = torch.nn.functional.pad(
            item["input_features"],
            (0, max_audio_len - item["input_features"].shape[1])
        )
        input_features.append(padded_audio)
        
        # 텍스트 패딩
        padded_text = torch.nn.functional.pad(
            item["decoder_input_ids"],
            (0, max_text_len - item["decoder_input_ids"].shape[0]),
            value=processor.tokenizer.pad_token_id
        )
        decoder_input_ids.append(padded_text)
        
        # 어텐션 마스크 패딩
        padded_mask = torch.nn.functional.pad(
            item["decoder_attention_mask"],
            (0, max_text_len - item["decoder_attention_mask"].shape[0])
        )
        decoder_attention_mask.append(padded_mask)
        
        # 라벨 패딩
        padded_labels = torch.nn.functional.pad(
            item["labels"],
            (0, max_text_len - item["labels"].shape[0]),
            value=-100  # loss 계산 시 무시될 패딩 값
        )
        labels.append(padded_labels)
    
    return {
        "input_features": torch.stack(input_features),
        "decoder_input_ids": torch.stack(decoder_input_ids),
        "decoder_attention_mask": torch.stack(decoder_attention_mask),
        "labels": torch.stack(labels)
    }

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # 데이터를 디바이스로 이동
        input_features = batch["input_features"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 모델 forward pass
        outputs = model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
        
        # Loss 계산
        loss = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Backward pass 및 최적화
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        # WandB에 로그 기록
        wandb.log({
            "batch_loss": loss.item(),
            "epoch": epoch
        })
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating'):
            input_features = batch["input_features"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask
            )
            
            loss = F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(eval_loader)
    return avg_loss

def main():
    # WandB 초기화
    wandb.init(project="whisper-llama-asr")
    
    # 모델과 프로세서 준비
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # 데이터셋 및 데이터로더 준비
    print("main train dataset preparing...")
    train_dataset = LibriSpeechDataset(split="train", processor=processor)
    print("main eval dataset preparing...")
    eval_dataset = LibriSpeechDataset(split="validation", processor=processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    modified_llama = modify_llama_blocks(llama, num_blocks_to_keep=2)
    whisper_encoder = whisper.model.encoder

    # 모델 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_asr_model(whisper_encoder, modified_llama)
    model = model.to(device)
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    # 학습 설정
    num_epochs = 1
    best_eval_loss = float('inf')
    
    # 학습 루프
    for epoch in range(num_epochs):
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # 평가
        eval_loss = evaluate(model, eval_loader, device)
        
        # WandB에 로그 기록
        wandb.log({
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "epoch": epoch
        })
        
        # 모델 저장
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pt')
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Eval Loss = {eval_loss:.4f}')

if __name__ == "__main__":
    main()