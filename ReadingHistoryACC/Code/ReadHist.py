import torch

checkpoint_path = "Checkpoint_PATH"

try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint['history']
    
    print(f"\n{'EPOCH':<8} | {'TRAIN ACC':<12} | {'VAL ACC':<12} | {'VAL LOSS':<10}")
    print("-" * 50)
    
    for i in range(len(history['train_acc'])):
        t_acc = history['train_acc'][i]
        v_acc = history['val_acc'][i]
        v_loss = history['val_loss'][i]
        print(f"{i+1:<8} | %{t_acc:<11.2f} | %{v_acc:<11.2f} | {v_loss:<10.4f}")
        
except Exception as e:
    print(f"Hata: {e}")