export interface RomResult {
  angle_rest: number;
  angle_dorsi: number;
  rom: number;
  image_rest: string; // base64 JPEG
  image_dorsi: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function calculateRom(rest: File, dorsi: File): Promise<RomResult> {
  const body = new FormData();
  body.append("image_rest", rest);
  body.append("image_dorsi", dorsi);

  const res = await fetch(`${API_URL}/api/calculate_rom`, { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}
