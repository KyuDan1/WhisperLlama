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
from torch.utils.checkpoint import checkpoint

from model import *

processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
whisper = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

class LibriSpeechDataset(Dataset):
    def __init__(self, split="train.100", whisper_processor=None, llama_tokenizer=None, max_samples=None):
        print("librispeech class load dataset...")
        self.dataset = load_dataset("fixie-ai/librispeech_asr", "clean", split=split, streaming=False)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(max_samples))
        self.whisper_processor = whisper_processor
        self.llama_tokenizer = llama_tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        audio = self.dataset[idx]["audio"]
        text = self.dataset[idx]["text"]
        
        # Whisper 프로세서로 오디오 특성 추출
        input_features = self.whisper_processor(
            audio["array"],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.squeeze(0)
        
        # Llama 토크나이저로 텍스트 토큰화
        labels = self.llama_tokenizer(
            text,
            padding="max_length",
            max_length=128,  # Llama의 sequence length에 맞춤
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_features": input_features,
            "labels": labels["input_ids"].squeeze(0)
        }

def collate_fn(batch):
    # Find max lengths in the batch
    max_audio_len = min(max(x["input_features"].shape[1] for x in batch), 480000)  # 30초로 제한
    max_text_len = min(max(x["labels"].shape[0] for x in batch), 128)  # 128 토큰으로 제한
    
    input_features = []
    labels = []
    
    for item in batch:
        # Truncate if longer than max length
        audio_features = item["input_features"][:, :max_audio_len]
        text_labels = item["labels"][:max_text_len]
        
        # Pad if shorter
        padded_audio = F.pad(
            audio_features,
            (0, max_audio_len - audio_features.shape[1])
        )
        padded_labels = F.pad(
            text_labels,
            (0, max_text_len - text_labels.shape[0]),
            value=-100
        )
        
        input_features.append(padded_audio)
        labels.append(padded_labels)
    
    return {
        "input_features": torch.stack(input_features),
        "labels": torch.stack(labels)
    }

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass with labels for teacher forcing
        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        current_loss = loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
        
        # Log to wandb
        wandb.log({
            "batch_loss": current_loss,
            "epoch": epoch,
            "step": batch_idx + epoch * len(train_loader)
        })
        
        # Optional: Generate sample predictions periodically
        if batch_idx % 100 == 0:
            model.eval()
            with torch.no_grad():
                sample_output = model(input_features=input_features[:1])  # Take first sample
                predicted_ids = torch.argmax(sample_output.logits, dim=-1)
                predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                actual_text = tokenizer.decode(labels[0], skip_special_tokens=True)
                
                wandb.log({
                    "sample_prediction": predicted_text,
                    "sample_actual": actual_text,
                    "step": batch_idx + epoch * len(train_loader)
                })
            model.train()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating'):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            
            # Store loss
            total_loss += loss.item()
            
            # Get predictions
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            
            # Convert to text for metrics
            for pred, label in zip(predicted_ids, labels):
                pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                label_text = tokenizer.decode(label, skip_special_tokens=True)
                
                all_predictions.append(pred_text)
                all_labels.append(label_text)
    
    # Calculate average loss
    avg_loss = total_loss / len(eval_loader)
    
    # Calculate WER (Word Error Rate) using basic string matching
    # You might want to use a proper WER calculation library for better results
    total_wer = 0
    for pred, label in zip(all_predictions, all_labels):
        pred_words = pred.split()
        label_words = label.split()
        
        # Calculate Levenshtein distance
        distance = levenshtein_distance(pred_words, label_words)
        wer = distance / len(label_words) if label_words else 0
        total_wer += wer
    
    avg_wer = total_wer / len(all_predictions) if all_predictions else 0
    
    # Log evaluation metrics
    wandb.log({
        "eval_loss": avg_loss,
        "eval_wer": avg_wer,
        "eval_samples": {
            "predictions": all_predictions[:5],  # Log first 5 samples
            "ground_truth": all_labels[:5]
        }
    })
    
    return avg_loss, avg_wer

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two word lists."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def main():
    torch.cuda.empty_cache()
    # WandB 초기화
    wandb.init(project="whisper-llama-asr")
    
    # 모델과 프로세서 준비
    
    #processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    whisper_processor = processor
    llama_tokenizer = tokenizer
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Llama 토크나이저 padding 설정
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
    
    # 패딩 방향 설정
    llama_tokenizer.padding_side = "right"
    
    # 데이터셋 및 데이터로더 준비
    print("main train dataset preparing...")
    train_dataset = LibriSpeechDataset(
        split="train.100", 
        whisper_processor=whisper_processor,
        llama_tokenizer=llama_tokenizer,
        max_samples=1000
    ) # 예시로 1000개로 제한
    print("main eval dataset preparing...")
    eval_dataset = LibriSpeechDataset(
        split="validation", 
        whisper_processor=whisper_processor,
        llama_tokenizer=llama_tokenizer,
        max_samples=100
    )
     # Llama 모델에도 pad_token_id 설정
    modified_llama = modify_llama_blocks(llama, num_blocks_to_keep=2)
    modified_llama.config.pad_token_id = llama_tokenizer.pad_token_id

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
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
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluation
        eval_loss, eval_wer = evaluate(model, eval_loader, device)
        
        # Save best model
        if eval_wer < best_wer:
            best_wer = eval_wer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'eval_wer': eval_wer,
            }, f'best_model_epoch_{epoch}_wer_{eval_wer:.4f}.pt')
        
        print(f'Epoch {epoch}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Eval Loss: {eval_loss:.4f}')
        print(f'  Eval WER: {eval_wer:.4f}')

if __name__ == "__main__":
    main()