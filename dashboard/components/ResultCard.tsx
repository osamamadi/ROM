import type { RomResult } from "@/lib/api";

interface AngleFigureProps {
  title: string;
  caption: string;
  angle: number;
  image: string; // base64 JPEG
  tone: "blue" | "slate";
}

function AngleFigure({ title, caption, angle, image, tone }: AngleFigureProps) {
  const badge =
    tone === "blue"
      ? "bg-blue-600 text-white shadow-blue-600/40"
      : "bg-slate-800 text-white shadow-slate-800/40";

  return (
    <figure className="flex flex-col items-center">
      <div className="w-full overflow-hidden rounded-2xl bg-slate-900 shadow-2xl shadow-slate-900/25 ring-1 ring-slate-900/10">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/jpeg;base64,${image}`}
          alt={`${title} with detected keypoints`}
          className="block h-auto w-full"
        />
      </div>
      <figcaption className="relative z-10 -mt-7 flex flex-col items-center">
        <span
          className={`inline-flex items-baseline gap-3 rounded-full px-7 py-3 ring-4 ring-white shadow-xl ${badge}`}
          aria-label={`${title} angle ${angle.toFixed(1)} degrees`}
        >
          <span className="text-xs font-semibold uppercase tracking-widest opacity-80">{title}</span>
          <span className="text-3xl font-extrabold tabular-nums">{angle.toFixed(1)}°</span>
        </span>
        <span className="mt-3 text-sm text-slate-500">{caption}</span>
      </figcaption>
    </figure>
  );
}

export default function ResultCard({ result }: { result: RomResult }) {
  return (
    <article className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-xl shadow-slate-900/5">
      {/* Report header */}
      <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50 px-6 py-3 sm:px-8">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
          <span className="h-2 w-2 rounded-full bg-emerald-500" />
          Ankle ROM Report
        </div>
        <span className="text-xs text-slate-400">
          {new Date().toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })}
        </span>
      </div>

      {/* Headline result */}
      <div className="px-6 pb-2 pt-12 text-center sm:px-8">
        <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">ROM Delta</p>
        <p className="mt-2 text-8xl font-black leading-none tracking-tight text-emerald-600 tabular-nums sm:text-9xl">
          {result.rom_deg.toFixed(1)}
          <span className="align-top text-5xl text-emerald-500 sm:text-6xl">°</span>
        </p>
        <p className="mt-4 text-sm text-slate-500">
          Dorsiflexion angle{" "}
          <span className="font-semibold text-slate-700">{result.dorsi.angle_deg.toFixed(1)}°</span>
          {" − "}resting angle{" "}
          <span className="font-semibold text-slate-700">{result.rest.angle_deg.toFixed(1)}°</span>
          {" = "}
          <span className="font-semibold text-emerald-600">{result.rom_deg.toFixed(1)}°</span>
        </p>
      </div>

      {result.warnings.length > 0 && (
        <div role="alert" className="mx-6 mt-8 rounded-xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-800 sm:mx-10 lg:mx-12">
          <p className="font-semibold">Check the keypoint placement before relying on this result</p>
          <ul className="mt-1 list-disc pl-5">
            {result.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="mx-6 my-8 border-t border-dashed border-slate-200 sm:mx-8" />

      {/* Side-by-side annotated images */}
      <div className="grid gap-x-10 gap-y-14 px-6 pb-12 sm:px-10 lg:px-12 md:grid-cols-2">
        <AngleFigure
          title="Resting"
          caption="Foot axis vs. vertical · Resting position"
          angle={result.rest.angle_deg}
          image={result.rest.overlay}
          tone="slate"
        />
        <AngleFigure
          title="Dorsiflexion"
          caption="Foot axis vs. vertical · Dorsiflexion position"
          angle={result.dorsi.angle_deg}
          image={result.dorsi.overlay}
          tone="blue"
        />
      </div>

      <div className="border-t border-slate-100 bg-slate-50 px-6 py-3 text-center text-xs text-slate-400 sm:px-8">
        Angle = foot axis (heel* → 5th metatarsal) vs. the vertical through the malleolus, where heel* is
        placed directly below the malleolus at sole level. Research prototype — not a certified medical device.
      </div>
    </article>
  );
}
