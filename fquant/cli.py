"""
FQuant CLI: Production Command-Line Interface for LLM Quantization.
"""

import os
import sys
import argparse
import logging
from .engine import FQuantEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FQuant")

def main():
    parser = argparse.ArgumentParser(
        description="FQuant: High-Precision Post-Training LLM Quantization Framework by F-Labs"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    quant_parser = subparsers.add_parser("quantize", help="Quantize an LLM using DV-SSQ & KV-BSS")
    quant_parser.add_argument("--model", "-m", type=str, required=True, help="Hugging Face repo ID or local directory path")
    quant_parser.add_argument("--output", "-o", type=str, required=True, help="Directory to store quantized safetensors model")
    quant_parser.add_argument("--group-size", "-g", type=int, default=64, help="Group size for INT4 GSQ (default: 64)")
    quant_parser.add_argument("--default-rank", "-r", type=int, default=16, help="Default SVD residual rank (default: 16)")
    quant_parser.add_argument("--k-proj-rank", type=int, default=32, help="SVD rank for key projections (default: 32)")
    quant_parser.add_argument("--device", type=str, default="cpu", help="Device to compute SVD (cpu or cuda)")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect model architecture and quantization suitability")
    inspect_parser.add_argument("--model", "-m", type=str, required=True, help="Hugging Face repo ID or local directory")

    verify_parser = subparsers.add_parser("verify", help="Verify integrity and reconstruction error of quantized model")
    verify_parser.add_argument("--model", "-m", type=str, required=True, help="Quantized model directory")

    args = parser.parse_args()

    if args.command == "quantize":
        logger.info("Initializing FQuant Engine...")
        engine = FQuantEngine(
            group_size=args.group_size,
            default_rank=args.default_rank,
            k_proj_rank=args.k_proj_rank,
            device=args.device
        )
        engine.quantize_model(model_source=args.model, output_dir=args.output)

    elif args.command == "inspect":
        engine = FQuantEngine()
        engine.inspect_model(args.model)

    elif args.command == "verify":
        engine = FQuantEngine()
        engine.verify_model(args.model)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
