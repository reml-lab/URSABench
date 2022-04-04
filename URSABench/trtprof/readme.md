# Profiling on Jetson Platform

1. Save PyTorch models to checkpoints to state dict files (`pt` or `pth` files).
2. Export `pt` or `pth` files to ONNX format:
   ```bash
   bash batch_torch2onnx.sh
   ```
3. Export ONNX models to TensorRT engines:
   ```bash
   bash batch_onnx2trt.sh
   ```
4. Profile TensorRT engines:
   ```bash
   bash pred.bash
   ```
