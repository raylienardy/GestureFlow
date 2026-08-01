const PER_HAND = 63;
const MAX_HANDS = 2;
const FEAT_DIM = PER_HAND * MAX_HANDS;

export function landmarksToFeature(multiHandLandmarks) {
  const feature = new Array(FEAT_DIM).fill(0);
  if (!multiHandLandmarks || multiHandLandmarks.length === 0) {
    return feature;
  }

  // Urutkan tangan berdasarkan wrist.x (kiri ke kanan)
  const hands = [...multiHandLandmarks];
  hands.sort((a, b) => a[0].x - b[0].x);

  const numHands = Math.min(hands.length, MAX_HANDS);

  for (let h = 0; h < numHands; h++) {
    const hand = hands[h];
    const arr = [];

    for (let i = 0; i < 21; i++) {
      // LANGSUNG gunakan nilai asli (kamera sudah mirror via CSS)
      const x = hand[i].x;
      const y = hand[i].y;
      const z = hand[i].z;
      arr.push(x, y, z);
    }

    // Translasi: kurangi wrist
    const wristX = arr[0];
    const wristY = arr[1];
    const wristZ = arr[2];
    for (let i = 0; i < 63; i += 3) {
      arr[i] -= wristX;
      arr[i + 1] -= wristY;
      arr[i + 2] -= wristZ;
    }

    // Skala
    let maxDist = 0;
    for (let i = 0; i < 21; i++) {
      const dx = arr[i * 3];
      const dy = arr[i * 3 + 1];
      const dz = arr[i * 3 + 2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist > maxDist) maxDist = dist;
    }
    if (maxDist > 0) {
      for (let i = 0; i < 63; i++) {
        arr[i] /= maxDist;
      }
    }

    const offset = h * PER_HAND;
    for (let i = 0; i < 63; i++) {
      feature[offset + i] = arr[i];
    }
  }
  return feature;
}
