"use client";

import { useEffect, useRef, useState } from "react";
import Dropzone from "@/components/Dropzone";
import ResultCard from "@/components/ResultCard";
import { calculateRom, type RomResult } from "@/lib/api";

export default function Home() {
  const [rest, setRest] = useState<File | null>(null);
  const [dorsi, setDorsi] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RomResult | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const ready = !!rest && !!dorsi && !loading;

  useEffect(() => {
    if (result) resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [result]);

  async function handleCalculate() {
    if (!rest || !dorsi) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await calculateRom(rest, dorsi));
    } catch (e) {
      setError(
        e instanceof TypeError
          ? "Could not reach the analysis server. Please try again shortly."
          : e instanceof Error
            ? e.message
            : "Something went wrong.",
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setRest(null);
    setDorsi(null);
    setResult(null);
    setError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50/70 via-slate-50 to-slate-50">
      {/* Top bar */}
      <nav className="border-b border-slate-200/70 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-600 text-white shadow-md shadow-blue-600/30">
              <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 20 12 4" />
                <path d="M4 20h16" />
                <path d="M9 20a6 6 0 00-1.5-4" />
              </svg>
            </span>
            <span className="text-lg font-bold tracking-tight text-slate-900">
              ROM<span className="text-blue-600">Vision</span>
            </span>
          </div>
          <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 ring-1 ring-emerald-200">
            Proof of Concept
          </span>
        </div>
      </nav>

      <main className="mx-auto max-w-5xl px-6 py-12">
        <header className="mx-auto mb-12 max-w-2xl text-center">
          <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl">
            Ankle Range of Motion,{" "}
            <span className="bg-gradient-to-r from-blue-600 to-emerald-500 bg-clip-text text-transparent">
              measured by AI
            </span>
          </h1>
          <p className="mt-4 text-lg text-slate-600">
            Upload a resting and a dorsiflexion photo. Our pose model locates the heel, 5th metatarsal and shin,
            then computes the angle change.
          </p>
        </header>

        <section className="grid gap-8 md:grid-cols-2">
          <Dropzone
            step={1}
            label="Resting Position"
            hint="Lateral view · foot relaxed · JPG, PNG or WebP"
            file={rest}
            onChange={setRest}
            disabled={loading}
          />
          <Dropzone
            step={2}
            label="Dorsiflexion Position"
            hint="Same view & framing · foot fully dorsiflexed"
            file={dorsi}
            onChange={setDorsi}
            disabled={loading}
          />
        </section>

        <div className="mt-10 flex flex-col items-center gap-4">
          <button
            type="button"
            onClick={handleCalculate}
            disabled={!ready}
            className="flex w-full max-w-lg items-center justify-center gap-3 rounded-2xl bg-blue-600 px-10 py-5 text-xl font-bold text-white shadow-xl shadow-blue-600/30 transition hover:bg-blue-700 hover:shadow-blue-700/30 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-blue-300 active:scale-[0.99] disabled:cursor-not-allowed disabled:bg-slate-300 disabled:shadow-none"
          >
            {loading && (
              <svg className="h-6 w-6 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden>
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
                <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
              </svg>
            )}
            {loading ? "Analyzing images…" : "Calculate ROM"}
          </button>

          {!ready && !loading && (
            <p className="text-sm text-slate-500">
              {!rest && !dorsi
                ? "Add both images to continue."
                : !rest
                  ? "Add the resting image to continue."
                  : "Add the dorsiflexion image to continue."}
            </p>
          )}

          {error && (
            <div role="alert" className="flex max-w-lg items-start gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <svg className="mt-0.5 h-4 w-4 shrink-0" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-4a1 1 0 00-1 1v3a1 1 0 002 0V7a1 1 0 00-1-1zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
              </svg>
              <span>{error}</span>
            </div>
          )}
        </div>

        {result && (
          <section ref={resultRef} className="mt-16 scroll-mt-8">
            <ResultCard result={result} />
            <div className="mt-6 text-center">
              <button
                type="button"
                onClick={reset}
                className="rounded-xl border border-slate-300 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-blue-400 hover:text-blue-700"
              >
                Start new measurement
              </button>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
