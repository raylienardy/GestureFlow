const SEQ_LEN = 30;

export class GestureClassifier {
  constructor() {
    this.model = null;
    this.labels = [];
    this.buffer = [];
  }

  async init(modelPath = "model/model.json", labelsPath = "model/labels.json") {
    const resp = await fetch(labelsPath);
    const labelObj = await resp.json();
    this.labels = Object.values(labelObj);
    this.model = await tf.loadLayersModel(modelPath);
    console.log("Model & labels loaded", this.labels);
  }

  addFrame(featureArray) {
    this.buffer.push(featureArray);
    while (this.buffer.length > SEQ_LEN) {
      this.buffer.shift();
    }
  }

  isReady() {
    return this.buffer.length === SEQ_LEN;
  }

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

  predictFromSequence(sequence) {
    if (!this.model) return null;
    const tensor = tf.tensor3d([sequence], [1, SEQ_LEN, 126]);
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
