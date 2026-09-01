"""Orchestrator for Model 3: build KG -> train+encode all 21 -> Layer-2 -> figures."""
import sys
from build_kg import main as build_det
from build_kg_llm import main as build_llm
from merge_kg import main as merge
from train_kg import main as train
from layer2.knn_shift_kg import main as layer2


def main():
    print("=== STEP 1: build deterministic KG ===", flush=True)
    build_det()
    print("\n=== STEP 2: build LLM KG (optional, skips if no API key) ===", flush=True)
    build_llm()
    print("\n=== STEP 3: merge KGs ===", flush=True)
    merge()
    print("\n=== STEP 4: train + encode (21 models, retrained per injection) ===", flush=True)
    train()
    print("\n=== STEP 5: Layer-2 KNN-shift + figures ===", flush=True)
    layer2()
    print("\n=== Model 3 complete ===", flush=True)


if __name__ == "__main__":
    main()
