# Dataset

`imdb_preprocessed_data.zip` contains the four arrays used by the notebook:

- `x_train.npy`: `(31818, 200)`, `int64`
- `y_train.npy`: `(31818,)`, `int64`
- `x_val.npy`: `(13636, 200)`, `int64`
- `y_val.npy`: `(13636,)`, `int64`

Run the following command from the repository root before opening the notebook:

```bash
python scripts/prepare_data.py
```

Archive SHA-256:

```text
df6be1272b0a773fb7a89c5f158fa3c96bd425b7ba4f882434ea0a3b3937e819
```

The arrays are course-provided, preprocessed data derived from the IMDB sentiment-classification dataset. Their upstream preprocessing script and redistribution license were not included in the original submission. Keep this repository private unless redistribution permission has been confirmed.

