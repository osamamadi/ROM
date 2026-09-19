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
      ? "bg-blue-600 text-white shadow-blue-600/25"
      : "bg-slate-800 text-white shadow-slate-800/25";

  return (
    <figure className="flex flex-col">
      <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/jpeg;base64,${image}`}
          alt={`${title} with detected keypoints`}
          className="h-80 w-full object-contain"
        />
      </div>
      <figcaption className="-mt-5 flex flex-col items-center">
        <span
          className={`inline-flex items-baseline gap-2 rounded-full px-5 py-2 shadow-lg ${badge}`}
          aria-label={`${title} angle ${angle.toFixed(1)} degrees`}
        >
          <span className="text-xs font-semibold uppercase tracking-widest opacity-80">{title}</span>
          <span className="text-2xl font-bold tabular-nums">{angle.toFixed(1)}°</span>
        </span>
        <span className="mt-2 text-xs text-slate-500">{caption}</span>
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
      <div className="px-6 pb-2 pt-10 text-center sm:px-8">
        <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">ROM Delta</p>
        <p className="mt-2 text-8xl font-black leading-none tracking-tight text-emerald-600 tabular-nums sm:text-9xl">
          {result.rom.toFixed(1)}
          <span className="align-top text-5xl text-emerald-500 sm:text-6xl">°</span>
        </p>
        <p className="mt-4 text-sm text-slate-500">
          Dorsiflexion angle{" "}
          <span className="font-semibold text-slate-700">{result.angle_dorsi.toFixed(1)}°</span>
          {" − "}resting angle{" "}
          <span className="font-semibold text-slate-700">{result.angle_rest.toFixed(1)}°</span>
          {" = "}
          <span className="font-semibold text-emerald-600">{result.rom.toFixed(1)}°</span>
        </p>
      </div>

      <div className="mx-6 my-8 border-t border-dashed border-slate-200 sm:mx-8" />

      {/* Side-by-side annotated images */}
      <div className="grid gap-x-8 gap-y-10 px-6 pb-10 sm:px-8 md:grid-cols-2">
        <AngleFigure
          title="Resting"
          caption="Angle at heel · Resting position"
          angle={result.angle_rest}
          image={result.image_rest}
          tone="slate"
        />
        <AngleFigure
          title="Dorsiflexion"
          caption="Angle at heel · Dorsiflexion position"
          angle={result.angle_dorsi}
          image={result.image_dorsi}
          tone="blue"
        />
      </div>

      <div className="border-t border-slate-100 bg-slate-50 px-6 py-3 text-center text-xs text-slate-400 sm:px-8">
        Angle measured at the heel between the 5th metatarsal and shin keypoints. Research prototype — not a
        certified medical device.
      </div>
    </article>
  );
}
