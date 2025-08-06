import argparse

from run_dynamic_benchmark import run_dynamic_benchmark
from run_static_benchmark import run_static_benchmark


def compute_ama_score(static_f1_task1, static_f1_task2, dynamic_acc_task1, dynamic_acc_task2, static_mae):
    """
    Computes a single AMA score based on static and dynamic benchmark results.
    The results are stored in a dictionary and printed out.
    """
    if None in [static_f1_task1, static_f1_task2, dynamic_acc_task1, dynamic_acc_task2, static_mae]:
        return None
    static_f1_avg = (static_f1_task1 + static_f1_task2) / 2 * 100
    dynamic_acc_avg = (dynamic_acc_task1 + dynamic_acc_task2) / 2 * 100
    max_reasonable_mae = 1.5  # Cap for normalising MAE
    penalty_weight = 10       # Max penalty ~10 points
    mae_penalty = min(static_mae / max_reasonable_mae, 1.0) * penalty_weight
    ama = 0.5 * static_f1_avg + 0.5 * dynamic_acc_avg - mae_penalty
    return round(ama, 2)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark tests")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--data-path", type=str, default="./benchmark/data", help="Path to the data directory")
    parser.add_argument("--dynamic-benchmark-model", type=str, default="alessioGalatolo/AMAEval", help="Path or (hf) repo to the model used for the dynamic benchmark.")
    args = parser.parse_args()

    print(f"Running benchmarks for model: {args.model}")
    print("Starting static benchmark...")
    static_results = run_static_benchmark(
        model=args.model,
        few_shot=True,
        data_path=args.data_path,
    )
    print("Static benchmark completed.")
    print("Starting dynamic benchmark...")
    dynamic_results = run_dynamic_benchmark(
        model=args.model,
        data_path=args.data_path,
        benchmark_model=args.dynamic_benchmark_model,
    )
    print("Dynamic benchmark completed.")

    static_f1_task1 = static_results["Static F1 Task1"]
    static_f1_task2 = static_results["Static F1 Task2"]
    dynamic_acc_task1 = dynamic_results["Dynamic Accuracy Task1"]
    dynamic_acc_task2 = dynamic_results["Dynamic Accuracy Task2"]
    static_mae = static_results["Static MAE Task1"]
    ama_score = compute_ama_score(
        static_f1_task1,
        static_f1_task2,
        dynamic_acc_task1,
        dynamic_acc_task2,
        static_mae
    )
    print(f"AMA Score for {args.model}: {ama_score:.3f}")
    print("Detailed results:")
    for key, value in static_results.items():
        print(f"{key}: {value:.3f}")
    for key, value in dynamic_results.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} entries")
        else:
            print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
