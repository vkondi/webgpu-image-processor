# 🖼️ WebGPU Image Processor

A simple React application that demonstrates how to use **WebGPU** in the browser for image processing. This project converts an uploaded image to **grayscale** using the GPU for parallel computation.

## ⚡How it works
- The uploaded image is drawn onto a `<canvas>`.
-  Pixel data is passed to the GPU.
- A compute shader calculates grayscale values using:
```
let gray = u32((r + g + b) / 3.0);
```
- The GPU writes the result back, and the processed pixels are rendered in another canvas.

### ⚠️ Note: 
WebGPU is only supported in latest Chrome (and behind a flag in some other browsers).