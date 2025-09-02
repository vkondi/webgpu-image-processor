import { useRef, useState } from "react";

export default function WebGPUImageProcessor() {
  const [original, setOriginal] = useState(null);
  const [processed, setProcessed] = useState(null);
  const fileInputRef = useRef();

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
      // scale image if too large for viewport
      const maxSize = 500; // px
      let { width, height } = img;
      if (width > maxSize || height > maxSize) {
        const ratio = Math.min(maxSize / width, maxSize / height);
        width = Math.round(width * ratio);
        height = Math.round(height * ratio);
      }

      const scaledCanvas = document.createElement("canvas");
      scaledCanvas.width = width;
      scaledCanvas.height = height;
      scaledCanvas.getContext("2d").drawImage(img, 0, 0, width, height);

      const scaledImg = new Image();
      scaledImg.src = scaledCanvas.toDataURL();

      scaledImg.onload = async () => {
        setOriginal(scaledImg.src);
        const result = await runWebGPU(scaledImg);
        setProcessed(result);
      };
    };
  }

  async function runWebGPU(image) {
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(image, 0, 0);
    const { data } = ctx.getImageData(0, 0, image.width, image.height);

    const pixels = new Uint32Array(image.width * image.height + 2);
    pixels[0] = image.width;
    pixels[1] = image.height;

    for (let i = 0; i < image.width * image.height; i++) {
      const r = data[i * 4 + 0];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const a = data[i * 4 + 3];
      pixels[i + 2] = (a << 24) | (b << 16) | (g << 8) | r;
    }

    if (!navigator.gpu) {
      alert("WebGPU not supported on this browser.");
      return;
    }
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const inputBuffer = device.createBuffer({
      size: pixels.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(inputBuffer.getMappedRange()).set(pixels);
    inputBuffer.unmap();

    const outputBuffer = device.createBuffer({
      size: pixels.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const readBuffer = device.createBuffer({
      size: pixels.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const shaderCode = /* wgsl */ `
      @group(0) @binding(0) var<storage, read> inputPixels : array<u32>;
      @group(0) @binding(1) var<storage, read_write> outputPixels : array<u32>;

      @compute @workgroup_size(16, 16)
      fn main(@builtin(global_invocation_id) GlobalId : vec3<u32>) {
          let width = inputPixels[0];
          let height = inputPixels[1];
          let x = GlobalId.x;
          let y = GlobalId.y;

          if (x >= width || y >= height) {
              return;
          }

          let idx = y * width + x + 2u;
          let pixel = inputPixels[idx];

          let r : f32 = f32((pixel >> 0u) & 0xFFu);
          let g : f32 = f32((pixel >> 8u) & 0xFFu);
          let b : f32 = f32((pixel >> 16u) & 0xFFu);
          let a : u32 = (pixel >> 24u) & 0xFFu;

          let gray = u32((r + g + b) / 3.0);

          outputPixels[idx] = (a << 24u) | (gray << 16u) | (gray << 8u) | gray;
      }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    const computePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "main" },
    });

    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const workgroupSize = 16;
    const dispatchX = Math.ceil(image.width / workgroupSize);
    const dispatchY = Math.ceil(image.height / workgroupSize);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY);

    passEncoder.end();

    commandEncoder.copyBufferToBuffer(
      outputBuffer,
      0,
      readBuffer,
      0,
      pixels.byteLength
    );

    device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const copyArray = new Uint32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();

    const outputImageData = ctx.createImageData(image.width, image.height);
    for (let i = 0; i < image.width * image.height; i++) {
      const pixel = copyArray[i + 2];
      outputImageData.data[i * 4 + 0] = pixel & 0xff;
      outputImageData.data[i * 4 + 1] = (pixel >> 8) & 0xff;
      outputImageData.data[i * 4 + 2] = (pixel >> 16) & 0xff;
      outputImageData.data[i * 4 + 3] = (pixel >> 24) & 0xff;
    }

    const outputCanvas = document.createElement("canvas");
    outputCanvas.width = image.width;
    outputCanvas.height = image.height;
    outputCanvas.getContext("2d").putImageData(outputImageData, 0, 0);

    return outputCanvas.toDataURL();
  }

  return (
    <div className="p-4 space-y-4">
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleUpload}
        className="mb-4"
      />

      <div className="flex space-x-8 items-start">
        {original && (
          <div className="flex flex-col items-center">
            <h3 className="mb-2">Original</h3>
            <img
              src={original}
              alt="Original"
              className="max-w-[500px] max-h-[500px] object-contain border"
            />
          </div>
        )}

        {processed && (
          <div className="flex flex-col items-center">
            <h3 className="mb-2">Grayscale</h3>
            <img
              src={processed}
              alt="Grayscale"
              className="max-w-[500px] max-h-[500px] object-contain border"
            />
          </div>
        )}
      </div>
    </div>
  );
}
