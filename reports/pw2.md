# PW2 Evaluation

*Generated: 2026-05-24 12:54*

**Grade: 2.5/20.0**

## Issues

Missing files:
- preprocess

- All packages have versions (Missing versions: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, openpyxl, pyarrow)
- build_model(filepath) signature (Expected ['filepath'], got [])
- make_predictions(filepath) signature (Expected ['filepath'], got [])
- Functions 2-10 lines (Violations:
  house_prices/train.py:build_model=14lines
  house_prices/inference.py:make_predictions=20lines)
- flake8 passes (flake8 errors: house_prices/__init__.py:1:1: W391 blank line at end of file
house_prices/inference.py:4:1: E302 expected 2 blank lines, found 1
house_prices/train.py:7:1: E302 expected 2 blank lines, found 1
3)
- Type hints present (Missing type hints: house_prices/preprocess.py:preprocess_data(return), house_prices/preprocess.py:preprocess_data(df), house_prices/train.py:build_model(return), house_prices/inference.py:make_predictions(return))
- Docstrings present (Missing docstrings: house_prices/preprocess.py:preprocess_data, house_prices/train.py:build_model, house_prices/inference.py:make_predictions)
- build_model returns dict[str, float] (TypeError: build_model() takes 0 positional arguments but 1 was given)
- make_predictions returns array (TypeError: make_predictions() takes 0 positional arguments but 1 was given)
- Artifacts saved after training (TypeError: build_model() takes 0 positional arguments but 1 was given)
- Artifacts loaded in inference (TypeError: make_predictions() takes 0 positional arguments but 1 was given)
- No select_dtypes (Forbidden: house_prices/preprocess.py: select_dtypes)
- pyproject.toml exists & installable (No pyproject.toml)
- Error handling for missing files (Expected FileNotFoundError, got TypeError: make_predictions() takes 0 positional arguments but 1 was given)
- Repository is private (Repository is public)
