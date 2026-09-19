"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface DropzoneProps {
  step: number;
  label: string;
  hint: string;
  file: File | null;
  onChange: (file: File | null) => void;
  disabled?: boolean;
}

const MAX_MB = 10;

export default function Dropzone({ step, label, hint, file, onChange, disabled }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept = useCallback(
    (f?: File | null) => {
      if (!f) return;
      if (!f.type.startsWith("image/")) return setLocalError("Please choose an image file (JPG, PNG, WebP).");
      if (f.size > MAX_MB * 1024 * 1024) return setLocalError(`Image must be smaller than ${MAX_MB} MB.`);
      setLocalError(null);
      onChange(f);
    },
    [onChange],
  );

  const open = () => !disabled && inputRef.current?.click();
  const done = !!file;

  return (
    <div className="flex flex-col">
      <div className="mb-3 flex items-center gap-3">
        <span
          className={`flex h-7 w-7 items-center justify-center rounded-full text-sm font-bold transition-colors ${
            done ? "bg-emerald-500 text-white" : "bg-blue-600 text-white"
          }`}
        >
          {done ? (
            <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M16.7 5.3a1 1 0 010 1.4l-7.5 7.5a1 1 0 01-1.4 0L3.3 9.7a1 1 0 011.4-1.4l3.8 3.8 6.8-6.8a1 1 0 011.4 0z" clipRule="evenodd" />
            </svg>
          ) : (
            step
          )}
        </span>
        <h2 className="text-base font-semibold text-slate-800">{label}</h2>
      </div>

      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label={`Upload ${label} image`}
        aria-disabled={disabled}
        onClick={open}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), open())}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) accept(e.dataTransfer.files?.[0]);
        }}
        className={[
          "group relative flex h-72 items-center justify-center overflow-hidden rounded-2xl border-2 outline-none transition-all",
          "focus-visible:ring-4 focus-visible:ring-blue-200",
          disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer",
          dragging
            ? "scale-[1.01] border-solid border-blue-500 bg-blue-50 shadow-lg shadow-blue-500/10"
            : done
              ? "border-solid border-emerald-300 bg-white shadow-sm"
              : "border-dashed border-slate-300 bg-white hover:border-blue-400 hover:bg-blue-50/40",
        ].join(" ")}
      >
        {preview ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt={`${label} preview`} className="h-full w-full bg-slate-50 object-contain" />
            <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-3 bg-gradient-to-t from-slate-900/70 to-transparent px-4 pb-3 pt-8">
              <span className="truncate text-xs font-medium text-white">{file?.name}</span>
              <div className="flex shrink-0 gap-2">
                <button
                  type="button"
                  disabled={disabled}
                  onClick={(e) => {
                    e.stopPropagation();
                    open();
                  }}
                  className="rounded-lg bg-white/90 px-3 py-1 text-xs font-semibold text-slate-700 hover:bg-white"
                >
                  Replace
                </button>
                <button
                  type="button"
                  disabled={disabled}
                  onClick={(e) => {
                    e.stopPropagation();
                    setLocalError(null);
                    onChange(null);
                  }}
                  className="rounded-lg bg-white/90 px-3 py-1 text-xs font-semibold text-red-600 hover:bg-white"
                >
                  Remove
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="pointer-events-none px-8 text-center">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-blue-50 text-blue-600 transition-transform group-hover:-translate-y-1">
              <svg className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.7}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0L8 8m4-4 4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
              </svg>
            </div>
            <p className="font-semibold text-slate-700">
              {dragging ? "Release to upload" : "Drag & drop an image"}
            </p>
            <p className="mt-1 text-sm text-slate-500">
              or <span className="font-medium text-blue-600 underline-offset-2 group-hover:underline">browse files</span>
            </p>
            <p className="mt-3 text-xs text-slate-400">{hint}</p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            accept(e.target.files?.[0]);
            e.target.value = ""; // allow re-selecting the same file
          }}
        />
      </div>

      {localError && (
        <p role="alert" className="mt-2 text-sm text-red-600">
          {localError}
        </p>
      )}
    </div>
  );
}
