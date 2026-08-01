export class HandDetector {
  constructor() {
    this.hands = null;
  }

  async init() {
    this.hands = new window.Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    await this.hands.initialize();
  }

  start(video, onResults) {
    this.hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    this.hands.onResults(onResults);
    const process = async () => {
      if (video.readyState >= 2) await this.hands.send({ image: video });
      requestAnimationFrame(process);
    };
    process();
  }

  stop() {
    if (this.hands) this.hands.close();
  }

  static draw(ctx, results, width, height) {
    const connections = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [0, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [0, 9],
      [9, 10],
      [10, 11],
      [11, 12],
      [0, 13],
      [13, 14],
      [14, 15],
      [15, 16],
      [0, 17],
      [17, 18],
      [18, 19],
      [19, 20],
    ];
    ctx.save();
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    for (const hand of results.multiHandLandmarks || []) {
      ctx.fillStyle = "#FF4081";
      for (const lm of hand) {
        ctx.beginPath();
        ctx.arc(lm.x * width, lm.y * height, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.strokeStyle = "#FF4081";
      ctx.lineWidth = 2;
      for (const [a, b] of connections) {
        ctx.beginPath();
        ctx.moveTo(hand[a].x * width, hand[a].y * height);
        ctx.lineTo(hand[b].x * width, hand[b].y * height);
        ctx.stroke();
      }
    }
    ctx.restore();
  }
}
