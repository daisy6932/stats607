"""Compare regenerated numerical results against the committed reference."""
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def compare(actual_path, reference_path):
    """Require identical methods/counts and close finite numerical summaries."""
    actual = pd.read_csv(actual_path).sort_values("method").reset_index(drop=True)
    reference = pd.read_csv(reference_path).sort_values("method").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual[["method", "n"]], reference[["method", "n"]], check_exact=True)
    pd.testing.assert_frame_equal(
        actual, reference, check_exact=False, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path,
                        default=ROOT / "results/tables/policy_comparison.csv")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "results/reference/policy_comparison.csv")
    args = parser.parse_args()
    try:
        compare(args.actual, args.reference)
    except (AssertionError, OSError, KeyError, ValueError) as error:
        raise SystemExit(f"Reference comparison failed: {error}") from error
    print("PASS: regenerated table matches the reference (rtol=1e-5, atol=1e-7).")
