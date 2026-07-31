/**
 * preprocessor.js
 * Meniru preprocessing Python: hands_to_feature().
 */

const PER_HAND = 63; // 21 titik * (x,y,z)
const MAX_HANDS = 2;
const FEAT_DIM = PER_HAND * MAX_HANDS; // 126

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

    // 1. Konversi ke array [x0,y0,z0, x1,y1,z1, ...]
    for (let i = 0; i < 21; i++) {
      arr.push(hand[i].x, hand[i].y, hand[i].z);
    }

    // 2. Translasi: kurangi wrist (landmark 0)
    const wristX = arr[0];
    const wristY = arr[1];
    const wristZ = arr[2];
    for (let i = 0; i < 63; i += 3) {
      arr[i] -= wristX;
      arr[i + 1] -= wristY;
      arr[i + 2] -= wristZ;
    }

    // 3. Hitung skala: max Euclidean distance
    let maxDist = 0;
    for (let i = 0; i < 21; i++) {
      const dx = arr[i * 3];
      const dy = arr[i * 3 + 1];
      const dz = arr[i * 3 + 2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist > maxDist) maxDist = dist;
    }

    // 4. Normalisasi skala
    if (maxDist > 0) {
      for (let i = 0; i < 63; i++) {
        arr[i] /= maxDist;
      }
    }

    // 5. Masukkan ke feature
    const offset = h * PER_HAND;
    for (let i = 0; i < 63; i++) {
      feature[offset + i] = arr[i];
    }
  }

  return feature;
}
