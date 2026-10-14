"""
preprocess.py
=============
Template for the ML Assignment preprocessing pipeline.

Instructions
------------
1. Implement your preprocessing steps in the `preprocess()` function.
2. Do NOT change the input/output filenames or the overall structure.


       python preprocess.py

   It will read  X_test.csv  and write  X_test_preprocessed.csv
   in the same directory.

Notes
-----
- Any parameters you fitted on the training set (e.g. mean, std, min, max)
  must be hard-coded here — do not load external files.
- The output must have the same number of rows as the input.
"""

import pandas as pd
import numpy as np

# =============================================================================
# PREPROCESSING FUNCTION — implement your pipeline here
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the raw feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw features loaded from X_test.csv.

    Returns
    -------
    pd.DataFrame
        Preprocessed features, ready for model inference.
    """

    df = df.copy()
    # Write your code below this line
    df = df.fillna(df.mean())

    # -------------------------------------------------------------------------
    # StandardScaler — parameters fitted on X_train.csv (hard-coded as required)
    # -------------------------------------------------------------------------
    # Computed via: StandardScaler().fit(X_train)
    means = {
        'x1': -6.50693740e-03,
        'x2':  1.85719890e+01,
        'x3':  2.50113890e+03,
        'x4':  5.05282981e-01,
    }
    stds = {
        'x1':  7.46067747e-01,
        'x2':  7.47720234e+02,
        'x3':  1.45141577e+03,
        'x4':  2.85221129e-01,
    }

    for col in ['x1', 'x2', 'x3', 'x4']:
        df[col] = (df[col] - means[col]) / stds[col]

    return df


# =============================================================================
# ENTRY POINT — do not modify below this line
# =============================================================================

if __name__ == "__main__":
    INPUT_FILE  = "X_test.csv"
    OUTPUT_FILE = "X_test_preprocessed.csv"

    print(f"Reading {INPUT_FILE} ...")
    raw = pd.read_csv(INPUT_FILE)
    print(f"  Input shape : {raw.shape}")

    processed = preprocess(raw)
    print(f"  Output shape: {processed.shape}")

    processed.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}.")
