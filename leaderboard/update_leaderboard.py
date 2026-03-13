"""
update_leaderboard.py
=====================
Evaluates a team's submission and updates the leaderboard.

Usage:
    python update_leaderboard.py --submission submissions/TEAM_NAME.csv --team TEAM_NAME
"""

import argparse
import json
import os
import sys

# ── Fix import path ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from leaderboard_utils import evaluate_submission, print_leaderboard, update_leaderboard


def main(args):
    print(f"\n{'='*55}")
    print(f"  DermAI Challenge — Submission Evaluation")
    print(f"{'='*55}")
    print(f"  Team       : {args.team}")
    print(f"  Submission : {args.submission}")
    print()

    if not os.path.exists(args.submission):
        print(f"Submission file not found: {args.submission}")
        sys.exit(1)

    print("Evaluating submission...")
    try:
        metrics = evaluate_submission(
            submission_path  = args.submission,
            true_labels_path = args.true_labels,
        )
    except ValueError as e:
        print(f"Invalid submission: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)

    print(f"\n  Results for team '{args.team}':")
    print(f"     F1-Score  (Macro) : {metrics['f1_score']:.4f}  (main metric)")
    print(f"     Accuracy          : {metrics['accuracy']:.4f}")
    print(f"     Precision (Macro) : {metrics['precision']:.4f}")
    print(f"     Recall    (Macro) : {metrics['recall']:.4f}")

    leaderboard_path = os.path.join(BASE_DIR, "leaderboard.csv")
    updated, improved = update_leaderboard(args.team, metrics, leaderboard_path)

    if improved:
        print(f"\n  Leaderboard updated — new best score for '{args.team}'!")
    else:
        print(f"\n  Previous score was better — leaderboard not updated.")

    print_leaderboard(updated)

    score_path = os.path.join(BASE_DIR, "score.json")
    with open(score_path, "w") as f:
        json.dump({"team": args.team, **metrics}, f, indent=2)
    print(f"  score.json saved\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermAI Challenge — Evaluate Submission")
    parser.add_argument("--submission",  required=True, help="Path to submission CSV")
    parser.add_argument("--team",        required=True, help="Team name")
    parser.add_argument("--true_labels",
                        default=os.path.join(BASE_DIR, "..", "data", "test_labels_TRUE.csv"),
                        help="Path to true labels CSV")
    main(parser.parse_args())
