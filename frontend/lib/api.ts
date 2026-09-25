export type KeypointName = "bottom_heel" | "5th_metatarsal" | "malleolus";

export interface ImageResult {
  angle_deg: number;
  /** [x, y] in ORIGINAL image pixel coordinates */
  keypoints: Record<KeypointName, [number, number]>;
  /** per-keypoint confidence, 0..1 */
  scores: Record<KeypointName, number>;
  /** base64 JPEG overlay (no data-URI prefix) */
  overlay: string;
}

export interface RomResult {
  rom_deg: number;
  rest: ImageResult;
  dorsi: ImageResult;
  warnings: string[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function calculateRom(rest: File, dorsi: File): Promise<RomResult> {
  const body = new FormData();
  body.append("image_rest", rest);
  body.append("image_dorsi", dorsi);

  const res = await fetch(`${API_URL}/api/calculate_rom`, { method: "POST", body });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let msg = text;
    try {
      const j = JSON.parse(text);
      msg = typeof j.detail === "string" ? j.detail : (j.error ?? JSON.stringify(j.detail ?? j));
    } catch {
      /* body was not JSON (e.g. a proxy 502 page): show it as-is */
    }
    throw new Error(`HTTP ${res.status}${res.statusText ? " " + res.statusText : ""}: ${msg.trim().slice(0, 300) || "(empty response body)"}`);
  }
  return res.json();
}
