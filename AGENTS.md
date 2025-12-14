# AGENTS.md

## Project Overview

추천 시스템 모델 구현 프로젝트. PyTorch 기반으로 전처리 → 훈련 → 평가 파이프라인을 제공한다.

## Directory Structure

```
src/
├── config.py              # CLI 설정 및 Config dataclass
├── models/
│   ├── registry.py        # MODEL_REGISTRY, LOSS_REGISTRY
│   ├── base_model.py      # BaseModel 추상 클래스
│   ├── loss.py            # 손실 함수들
│   ├── mf.py, lightgcn.py, sasrec.py  # 모델 구현
│   └── utils.py           # build_model()
├── process/
│   └── processor.py       # 데이터 전처리
└── train/
    ├── dataset.py         # PyTorch Dataset 클래스
    ├── trainer.py         # 훈련/평가 로직
    └── utils.py           # build_dataloaders()
```

## Key Patterns

### Registry Pattern

모델과 Loss는 데코레이터로 등록된다. 새 모델/Loss 추가 시 해당 파일만 수정하면 된다.

```python
# 새 모델 추가
@register_model("model_name", model_type="cf")  # or "sequential"
class NewModel(BaseModel):
    @classmethod
    def build(cls, config, data):
        return cls(...)
    
    def forward(self, ...): ...
    def calc_loss(self, ...): ...
    def recommend(self, ...): ...
```

```python
# 새 Loss 추가
@register_loss("loss_name")
class NewLoss(BaseLossWithNegativeSamples):
    def __call__(self, ufeats, pos_ifeats, neg_ifeats, pad_mask=None):
        ...
```

### Model Types

- `cf`: Collaborative Filtering (MF, LightGCN) - 유저/아이템 ID 기반
- `sequential`: Sequential Recommendation (SASRec) - 시퀀스 기반

`model_type`에 따라 Dataset 클래스가 자동 선택된다.

## Conventions

- Python 3.11+, PyTorch < 2.4.0
- Type hints 사용 (순환 import 시 `TYPE_CHECKING` 활용)
- `from __future__ import annotations` 사용
- Dataclass로 설정 관리
- 모든 모델은 `BaseModel` 상속, `build()` 클래스 메서드 구현 필수

## Adding New Components

### New Model

1. `src/models/new_model.py` 생성
2. `@register_model` 데코레이터 적용
3. `build()` 클래스 메서드 구현
4. `src/models/__init__.py`에 import 추가

### New Loss

1. `src/models/loss.py`에 클래스 추가
2. `@register_loss` 데코레이터 적용

### New Dataset

1. `src/process/processor.py`에 로더 함수 추가
2. `config.py`의 `--dataset` choices에 추가

## Commands

```bash
# 실행
uv run main.py --model mf --loss_fn ce

# 도움말
uv run main.py --help
```

