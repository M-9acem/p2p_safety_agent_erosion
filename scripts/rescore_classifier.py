"""Re-score an existing run's saved generations with the LLM-judge
classifier, and report agreement with the substring scorer (spec section
6: score both ways, report disagreement — don't collapse to one number).

Post-hoc utility over files run_experiment.py already wrote (each
stepN_generations.json / seedN_examples.json under a run's output_dir),
not a new Hydra-driven experiment — plain argparse, not hydra.main.

Usage:
    python scripts/rescore_classifier.py outputs/2026-09-04/20-19-58/single_agent
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="dir containing *_generations.json files")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--judge-dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--judge-batch-size", type=int, default=16)
    args = parser.parse_args()

    from p2p_safety.eval.safety import score_batch_classifier, score_batch_substring, scorer_agreement
    from p2p_safety.model_utils import batched_generate, load_model_and_tokenizer

    judge_model, judge_tokenizer = load_model_and_tokenizer(args.judge_model, args.judge_dtype, args.device)
    print(f"loaded judge model {args.judge_model}")

    def judge_generate(judge_prompts: list[str]) -> list[str]:
        return batched_generate(
            judge_model,
            judge_tokenizer,
            judge_prompts,
            max_new_tokens=8,
            do_sample=False,
            batch_size=args.judge_batch_size,
            device=args.device,
        )

    files = sorted(args.input_dir.glob("*_generations.json"))
    if not files:
        raise SystemExit(f"no *_generations.json files found in {args.input_dir}")

    report = {}
    for f in files:
        examples = json.loads(f.read_text())
        prompts = [e["prompt"] for e in examples]
        generations = [e["generation"] for e in examples]

        substring_result = score_batch_substring(generations)
        classifier_result = score_batch_classifier(generations, prompts, judge_generate)
        agreement = scorer_agreement(substring_result, classifier_result)

        checkpoint = f.stem.replace("_generations", "")
        report[checkpoint] = {
            "n_total": substring_result["n_total"],
            "substring_asr": substring_result["asr"],
            "classifier_asr": classifier_result["asr"],
            "classifier_n_unparseable": classifier_result["n_excluded_empty"] - substring_result["n_excluded_empty"],
            "scorer_agreement": agreement,
        }
        r = report[checkpoint]
        print(
            f"{checkpoint}: substring_asr={r['substring_asr']:.4f} "
            f"classifier_asr={r['classifier_asr']:.4f} agreement={r['scorer_agreement']:.4f} "
            f"(n_unparseable_judge_answers={r['classifier_n_unparseable']})"
        )

    out_path = args.input_dir / "classifier_rescore.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
