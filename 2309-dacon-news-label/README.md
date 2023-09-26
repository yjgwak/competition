# 월간 데이콘 쇼츠 - 뉴스 기사 레이블 복구 해커톤
## 문제 정의
라벨이 없는 60,000개의 뉴스 기사의 제목과 내용을 통해 6개의 카테고리로 분류
## 접근 방법
### 라벨링 기법
1. 첫 560개의 데이터를 직접 라벨링을 한 이후 사전 학습한 모델을 Fine-tuning
    - Public Macro F1 0.8248062383
2. 전체 데이터를 Pseudo Labeling을 거친 이후, 전체 데이터에 대해 학습 후 추론
    - Public Macro F1 0.8283695637
### 모델 선정 이유
유사한 데이터로 학습된 공개 사전 학습 모델을 활용 후 파인 튜닝.

### 코드 실행 방법
직접 라벨링한 560개의 데이터를 통한 Fine-tuning 및 Pseudo labeling
```bash
python train.py --news_data data/news.csv --submit_data data/sample_submission.csv --manual_data data/manual.csv --num_epochs 4 --output pseudo-labeling.csv
```

Pseudo labeling 결과를 활용해 전체 학습 및 추론
```bash
python train.py --news_data data/news.csv --submit_data data/sample_submission.csv --manual_data pseudo-labeling.csv --num_epochs 1 --output full-training.csv
```

### 개선 방법
- Pseudo labeling을 한 데이터 중 일부만 활용해 Fine-tuning
- 여러 모델을 활용한 Ensemble