# Recommendation Models

PyTorch 기반 추천 시스템 모델 구현 프로젝트.

## Features

- **Models**: Matrix Factorization, LightGCN, SASRec
- **Losses**: BPR, BCE, Cross Entropy
- **Dataset**: MovieLens-1M
- **Evaluation**: Recall@K, NDCG@K

## Installation

```bash
./init.bash <python_version>
```

> PyTorch 버전은 < 2.4.0

## Usage

```bash
# 기본 실행 (MF 모델)
uv run main.py

# 모델 선택
uv run main.py --model lightgcn
uv run main.py --model sasrec

# 손실 함수 선택
uv run main.py --loss_fn bpr

# 전체 옵션 확인
uv run main.py --help
```

## Project Structure

```
src/
├── config.py           # CLI 설정
├── models/             # 모델 구현
│   ├── registry.py     # Registry 패턴
│   ├── base_model.py   # 베이스 클래스
│   ├── loss.py         # 손실 함수
│   ├── mf.py           # Matrix Factorization
│   ├── lightgcn.py     # LightGCN
│   └── sasrec.py       # SASRec
├── process/            # 데이터 전처리
└── train/              # 훈련 로직
```

## Adding New Models

Registry 패턴을 사용하여 새 모델을 쉽게 추가할 수 있습니다:

```python
from src.models.base_model import BaseModel
from src.models.registry import register_model

@register_model("new_model", model_type="cf")
class NewModel(BaseModel):
    @classmethod
    def build(cls, config, data):
        return cls(...)
    
    def forward(self): ...
    def calc_loss(self, ...): ...
    def recommend(self, ...): ...
```

자세한 내용은 [AGENTS.md](./AGENTS.md)를 참조하세요.
