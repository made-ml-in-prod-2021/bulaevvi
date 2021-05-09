python reports/make_report.py
python ml_example/train_pipeline.py configs/config_lr.yml train
python ml_example/train_pipeline.py configs/config_rf.yml train
python ml_example/train_pipeline.py configs/config_lr.yml predict
python ml_example/train_pipeline.py configs/config_rf.yml predict
cd ml_example
pytest ../tests --cov
cd ..