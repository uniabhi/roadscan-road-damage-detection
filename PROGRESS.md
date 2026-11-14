# Progress Tracker

Update this file as you progress through the project.

## Week 1-2: Data Audit and Splits

- [ ] Create GitHub repo and README stub
- [ ] Download RDD2022 dataset on Kaggle
- [ ] Run audit.py and generate audit_summary.md
- [ ] Convert VOC to YOLO format
- [ ] Group near-duplicate images
- [ ] Create leakage-safe splits
- [ ] Save split lists and track with DVC

## Week 3-4: Training and Evaluation

- [ ] Run smoke test (2 epochs) to measure epoch time
- [ ] Train S1 multi-country model
- [ ] Train S1 variants (yolo11n vs s, imgsz 640 vs 960)
- [ ] Evaluate on global test set
- [ ] Generate per-country metrics table
- [ ] Log all runs to MLflow/DagsHub

## Week 5: Error Analysis and Severity

- [ ] Create failure gallery (20 FP, 20 FN, 10 TP)
- [ ] Run label-noise review
- [ ] Implement severity scoring
- [ ] Compute severity thresholds from validation
- [ ] Export model to ONNX
- [ ] Benchmark CPU latency

## Week 6: LLM Layer

- [ ] Implement LLM client interface
- [ ] Create prompt templates
- [ ] Build RAG retriever with repair guidelines
- [ ] Implement grounding checker
- [ ] Test with mock LLM
- [ ] Measure grounding pass rate on 50 reports

## Week 7: Serving and CI/CD

- [ ] Build FastAPI service
- [ ] Add /health and /detect endpoints
- [ ] Create Dockerfile
- [ ] Set up GitHub Actions CI
- [ ] Implement metric gate
- [ ] Deploy to AWS ECS Fargate
- [ ] Record demo video

## Week 8: Documentation and Polish

- [ ] Complete README with results
- [ ] Write project story document
- [ ] Prepare interview answers
- [ ] Final cleanup and testing
- [ ] Teardown AWS resources