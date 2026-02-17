# AGENTS.md

## 목적

추천 시스템 모델 구현 프로젝트. PyTorch 기반으로 전처리 → 훈련 → 평가 파이프라인을 제공한다.

## 핵심 구조

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

## 필수 규칙

- Python 버전은 `>=3.11, <3.12`를 사용한다.
- 모델/Loss는 `registry.py` 데코레이터로 등록한다.
- 모든 모델은 `BaseModel`을 상속하고 `build()` 클래스 메서드를 구현한다.
- 모델은 `forward()`, `calc_loss()`, `recommend()`를 구현한다.
- `model_type`에 따라 Dataset이 자동 선택된다.
- `cf`: Collaborative Filtering (MF, LightGCN) - 유저/아이템 ID 기반
- `sequential`: Sequential Recommendation (SASRec) - 시퀀스 기반

## 확장 절차

1. 새 모델: `src/models/`에 파일 추가 → `@register_model` 적용 → `build()` 구현 → `src/models/__init__.py` import 추가
2. 새 Loss: `src/models/loss.py`에 클래스 추가 → `@register_loss` 적용
3. 새 Dataset: `src/process/processor.py`에 로더 추가 → `config.py`의 `--dataset` choices 갱신

## 기본 명령

```bash
uv run main.py --model mf --loss_fn ce
uv run main.py --help
```

