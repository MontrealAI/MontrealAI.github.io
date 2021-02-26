async function runExample() {
  // Create an ONNX inference session with WebGL backend.
  const session = new onnx.InferenceSession({ backendHint: 'webgl' });

  // Load an ONNX model. This model is Resnet50 that takes a 1*3*299*299 image and classifies it.
  await session.loadModel("./resnetpaintings.onnx");

  // Load image.
  const imageLoader = new ImageLoader(imageSize, imageSize);
  const imageData = await imageLoader.getImageData('./art_nouveau-modern_quintessential_ai.jpg');

  // Preprocess the image data to match input dimension requirement, which is 1*3*299*299.
  const width = imageSize;
  const height = imageSize;
  const preprocessedData = preprocess(imageData.data, width, height);

  const inputTensor = new onnx.Tensor(preprocessedData, 'float32', [1, 3, width, height]);
  // Run model with Tensor inputs and get the result.
  const outputMap = await session.run([inputTensor]);
  const outputData = outputMap.values().next().value.data;
  const softMaxconst = softMax(outputData);
  const topK = PaintingClassesTopK(softMaxconst, 5);
  // const argMax = argMax(outputData);
  
  console.log("outputData = ", outputData)
  console.log("softMax = ", softMaxconst)
  console.log("topK = ", topK)
  // console.log("argMax = ", argMax)
  
  // console.log(
    // "outputData = ", outputData,
    // "softmax = ", probs)

  // Render the output result in html.
  printMatches(softMaxconst);
}

/**
 * Softmax function.
 */

function softMax(arr) {
  const C = Math.max(...arr);
  const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
  return arr.map((value, index) => { 
    return Math.exp(value - C) / d;
  })
}

/**
 * Preprocess raw image data to match Resnet requirement.
 */
function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);

  // Normalize 0-255 to (-1)-1
  ndarray.ops.divseq(dataFromImage, 128.0);
  ndarray.ops.subseq(dataFromImage, 1.0);

  // Realign imageData from [299*299*4] to the correct dimension [1*3*299*299].
  ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
  ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));

  return dataProcessed.data;
}

/**
 * Utility function to post-process Resnet output. Find top k classes with highest probability.
 */
function PaintingClassesTopK(outputData, k) {
  if (!k) { k = 5; }
  const probs = Array.from(outputData);
  const probsIndices = probs.map(
    function (prob, index) {
      return [prob, index];
    }
  );
  const sorted = probsIndices.sort(
    function (a, b) {
      if (a[0] < b[0]) {
        return -1;
      }
      if (a[0] > b[0]) {
        return 1;
      }
      return 0;
    }
  ).reverse();
  const topK = sorted.slice(0, k).map(function (probIndex) {
    return {
      index: parseInt(probIndex[1], 10),
      probability: probIndex[0]
    };
  });
  return topK;
}

/**
 * Render Resnet output to Html.
 */
function printMatches(data) {
  let outputClasses = [];
  if (!data || data.length === 0) {
    const empty = [];
    for (let i = 0; i < 5; i++) {
      empty.push({ name: '-', probability: 0, index: 0 });
    }
    outputClasses = empty;
  } else {
    outputClasses = PaintingClassesTopK(data, 5);
  }
  const predictions = document.getElementById('predictions');
  predictions.innerHTML = '';
  const results = [];
  for (let i of [0, 1, 2, 3, 4]) {
    results.push(`${outputClasses[i].name}: ${Math.round(100 * outputClasses[i].probability)}%`);
  }
  predictions.innerHTML = results.join('<br/>');
}

