/**
 * classifier.js
 * Memuat model LSTM dan menjalankan inferensi.
 */

const SEQ_LEN = 30;

export class GestureClassifier {
  constructor() {
    this.model = null;
    this.labels = [];
    this.buffer = [];
  }

  async init(modelPath = "model/model.json", labelsPath = "model/labels.json") {
    // Load labels
    const resp = await fetch(labelsPath);
    const labelObj = await resp.json(); // {"0":"halo","1":"salam"}
    this.labels = Object.values(labelObj);

    // Load model
    this.model = await tf.loadLayersModel(modelPath);
    console.log("Model & labels loaded", this.labels);
  }

  /**
   * Tambahkan satu frame (array 126) ke buffer.
   */
  addFrame(featureArray) {
    this.buffer.push(featureArray);
    // Jaga buffer tetap berukuran SEQ_LEN (sliding window)
    while (this.buffer.length > SEQ_LEN) {
      this.buffer.shift();
    }
  }

  isReady() {
    return this.buffer.length === SEQ_LEN;
  }

  /**
   * Jalankan inferensi, kembalikan { label, confidence }.
   */
  predict() {
    if (!this.model || !this.isReady()) return null;

    const tensor = tf.tensor3d([this.buffer], [1, SEQ_LEN, 126]);
    const output = this.model.predict(tensor);
    const scores = output.dataSync();
    tensor.dispose();
    output.dispose();

    const maxIdx = scores.indexOf(Math.max(...scores));
    return {
      label: this.labels[maxIdx],
      confidence: scores[maxIdx],
    };
  }
}
