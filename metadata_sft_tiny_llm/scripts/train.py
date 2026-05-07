import glob
import random

# 找出所有的 chunk 檔案
chunk_files = glob.glob("./top_k_chunks/chunk_*.pt")
random.shuffle(chunk_files) # 打亂順序

epochs = 3
global_step = 0

for epoch in range(epochs):
    print(f"=== 開始 Epoch {epoch+1} ===")
    
    for chunk_file in chunk_files:
        # 讀取這 10,000 筆的 chunk
        chunk_data = torch.load(chunk_file) 
        
        # 自己寫一個簡單的 Batch Generator
        batch_size = 16 # 根據 5070 Ti VRAM 調整
        for i in range(0, len(chunk_data), batch_size):
            batch = chunk_data[i : i+batch_size]
            
            # 把 List of Dict 轉成 Tensors，並送上 GPU
            input_ids = torch.stack([item["input_ids"] for item in batch]).to("cuda")
            top_k_val = torch.stack([item["top_k_values"] for item in batch]).to("cuda")
            top_k_idx = torch.stack([item["top_k_indices"] for item in batch]).to("cuda")
            
            # 1. 清空梯度
            optimizer.zero_grad()
            
            # 2. Forward Pass (讓 Tiny-LLM 預測)
            outputs = draft_model(input_ids)
            draft_logits = outputs.logits
            
            # 3. 計算 Loss
            # 動態 Alpha：前 1000 步先用 CE (alpha=1.0) 打底，之後慢慢降到 0.25
            current_alpha = 1.0 if global_step < 1000 else 0.25
            loss = compute_distillation_loss(
                draft_logits, top_k_val, top_k_idx, input_ids, alpha=current_alpha
            )
            
            # 4. Backward Pass & 更新權重
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # 監控指標
            if global_step % 100 == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f} | Alpha: {current_alpha}")
                
        # 每個 Chunk 跑完可以考慮存個檔
        print(f"完成 Chunk: {chunk_file}")
        
    # Epoch 結束存檔
    draft_model.save_pretrained(f"./draft_model_epoch_{epoch}")
    tokenizer.save_pretrained(f"./draft_model_epoch_{epoch}")