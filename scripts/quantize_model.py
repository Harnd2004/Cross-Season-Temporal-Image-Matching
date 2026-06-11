import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

def main():
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    # Path to original model
    cache_dir = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints"))
    model_fp32 = cache_dir / "superpoint_lightglue_fused_cpu.onnx"
    
    if not model_fp32.exists():
        logger.error(f"FP32 model not found at {model_fp32}. Please run the align_pairs.py script first to download the weights.")
        return

    # Path to output model
    output_dir = ROOT / "models"
    output_dir.mkdir(exist_ok=True)
    model_quant = output_dir / "superpoint_lightglue_fused_cpu_quantized.onnx"

    logger.info(f"Quantizing {model_fp32} -> {model_quant} ...")
    
    quantize_dynamic(
        model_input=str(model_fp32),
        model_output=str(model_quant),
        weight_type=QuantType.QInt8
    )
    
    logger.info("Quantization completed successfully!")
    logger.info(f"Original size: {model_fp32.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"Quantized size: {model_quant.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
