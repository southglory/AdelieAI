# Training Recipe

이 adapter는 differentia-llm `core.training.trainer.train_lora()`로 산출되었다.

## 입력

- base model: `models\upstream\Qwen2.5-7B-Instruct`
- 데이터셋: `core/training/dataset.py` ROLEPLAY_PAIRS (120건)

## 하이퍼파라미터

- num_epochs: 4
- learning_rate: 0.0002
- per_device_batch_size: 2
- gradient_accumulation_steps: 4
- max_seq_length: 1024
- LoRA r=16, alpha=32, dropout=0.05, target_modules=q/k/v/o/gate/up/down_proj

## 결과

- trainable params: 40,370,176 / 7,655,986,688 (0.5273%)
- final training loss: 0.9498
- wall clock: 121.3s

## 재현

```python
from core.training.trainer import train_lora
train_lora(base_model_path='models\\upstream\\Qwen2.5-7B-Instruct',
           output_dir='models\\ours\\qwen-roleplay-v2',
           num_epochs=4)
```
