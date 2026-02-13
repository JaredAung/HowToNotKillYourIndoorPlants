"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [apiStatus, setApiStatus] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/`)
      .then((res) => res.json())
      .then((data) => setApiStatus(data.message))
      .catch(() => setApiStatus("Backend not connected"));
  }, []);

  return (
    <div className="flex min-h-screen items-center justify-center bg-emerald-50 font-sans dark:bg-zinc-950">
      <main className="flex max-w-3xl flex-col items-center gap-8 px-8 py-16 text-center">
        <h1 className="text-4xl font-bold tracking-tight text-emerald-800 dark:text-emerald-100">
          How To Not Kill Your Indoor Plants
        </h1>
        <p className="text-lg text-zinc-600 dark:text-zinc-400">
          Your indoor plant care companion
        </p>
        <div className="rounded-lg bg-white px-6 py-4 shadow-sm dark:bg-zinc-900">
          <p className="text-sm text-zinc-500">API Status</p>
          <p className="font-medium text-emerald-600 dark:text-emerald-400">
            {apiStatus || "Checking..."}
          </p>
        </div>
      </main>
    </div>
  );
}
