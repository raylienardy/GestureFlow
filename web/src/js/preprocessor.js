const PER_HAND = 63; // 21 titik * 3 (x,y,z)
const MAX_HANDS = 2;
const FEAT_DIM = PER_HAND * MAX_HANDS; // 126

/**
 * Sama persis dengan landmarks_to_array() di train.py
 * - Tidak ada mirror (kamera sudah mirror via CSS)
 * - Tidak ada normalisasi wrist/scale
 * - Tangan diurutkan berdasarkan wrist.x
 * - Maksimal 2 tangan, sisanya diisi nol
 */
export function landmarksToFeature(multiHandLandmarks, multiHandedness) {
  const feature = new Array(FEAT_DIM).fill(0);
  if (!multiHandLandmarks || multiHandLandmarks.length === 0) return feature;

  let hands = [...multiHandLandmarks];
  if (multiHandedness && multiHandedness.length > 0) {
    const labels = multiHandedness.map((h) => h.label);
    const sortedIndices = labels
      .map((l, i) => [l, i])
      .sort((a, b) => (a[0] === "Left" ? 0 : 1) - (b[0] === "Left" ? 0 : 1))
      .map((x) => x[1]);
    hands = sortedIndices.map((i) => multiHandLandmarks[i]);
  } else {
    hands.sort((a, b) => a[0].x - b[0].x);
  }

  for (let h = 0; h < Math.min(hands.length, MAX_HANDS); h++) {
    const hand = hands[h];
    const offset = h * PER_HAND;
    for (let i = 0; i < 21; i++) {
      const lm = hand[i];
      const idx = offset + i * 3;
      feature[idx] = lm.x;
      feature[idx + 1] = lm.y;
      feature[idx + 2] = lm.z;
    }
  }
  return feature;
}
