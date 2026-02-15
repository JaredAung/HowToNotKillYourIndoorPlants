"use client";

import Link from "next/link";
import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type GardenPlant = { latin: string; name: string; plant_id: string; image_url: string };

export default function GardenPage() {
  const [username, setUsername] = useState("");
  const [inputUsername, setInputUsername] = useState("");
  const [plants, setPlants] = useState<GardenPlant[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchGarden = async (name: string) => {
    if (!name.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_URL}/api/garden?username=${encodeURIComponent(name.trim())}`
      );
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      setPlants(data.plants ?? []);
      setUsername(name.trim());
    } catch {
      setError("Could not load your garden. Is the backend running?");
      setPlants([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchGarden(inputUsername);
  };

  // No username entered yet - show prompt
  if (!username) {
    return (
      <div className="flex min-h-screen flex-col bg-white dark:bg-zinc-900">
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-md px-4 py-12">
            <h2 className="mb-2 text-xl font-semibold text-zinc-800 dark:text-zinc-200">
              My Garden
            </h2>
            <p className="mb-6 text-zinc-600 dark:text-zinc-400">
              Enter your username to view your garden plants.
            </p>
            <form onSubmit={handleSubmit} className="space-y-4">
              <input
                type="text"
                value={inputUsername}
                onChange={(e) => setInputUsername(e.target.value)}
                placeholder="Your username"
                className="w-full rounded-xl border border-zinc-200 bg-white px-4 py-3 text-zinc-800 placeholder:text-zinc-400 focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:placeholder:text-zinc-500"
                autoFocus
              />
              <button
                type="submit"
                disabled={loading || !inputUsername.trim()}
                className="w-full rounded-xl bg-emerald-600 px-4 py-3 font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
              >
                {loading ? "Loading..." : "View my garden"}
              </button>
            </form>
            {error && (
              <p className="mt-4 text-sm text-red-600 dark:text-red-400">
                {error}
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Username set - show garden with option to change
  return (
    <div className="flex min-h-screen flex-col bg-white dark:bg-zinc-900">
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 py-8">
          <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold text-zinc-800 dark:text-zinc-200">
                {username}&apos;s Garden
              </h2>
              <button
                onClick={() => {
                  setUsername("");
                  setPlants([]);
                  setError(null);
                }}
                className="mt-1 text-sm text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-300"
              >
                Use a different username
              </button>
            </div>
            {loading && (
              <span className="text-sm text-zinc-500">Refreshing...</span>
            )}
          </div>

          {error && (
            <p className="mb-4 text-sm text-red-600 dark:text-red-400">
              {error}
            </p>
          )}

          {plants.length === 0 && !loading ? (
            <p className="text-zinc-600 dark:text-zinc-400">
              No plants in your garden yet. Add plants from the Recommender by
              saying &quot;I&apos;ll take the first one&quot; or &quot;add the
              croton to my garden&quot;.
            </p>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {plants.map((plant, i) => {
                const CardContent = (
                  <>
                    <div className="aspect-square w-full overflow-hidden bg-zinc-200 dark:bg-zinc-600">
                      {plant.image_url ? (
                        <img
                          src={plant.image_url}
                          alt={plant.latin}
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <div className="flex h-full w-full items-center justify-center text-zinc-400 text-xs">
                          No image
                        </div>
                      )}
                    </div>
                    <p className="p-3 text-center font-medium text-emerald-700 dark:text-emerald-400">
                      {plant.latin}
                    </p>
                  </>
                );
                const cardClass =
                  "block overflow-hidden rounded-xl border border-zinc-200 bg-zinc-50 transition-shadow hover:shadow-md dark:border-zinc-700 dark:bg-zinc-800/50";
                return plant.plant_id ? (
                  <Link key={plant.plant_id || i} href={`/plant/${plant.plant_id}`} className={cardClass}>
                    {CardContent}
                  </Link>
                ) : (
                  <div key={i} className={cardClass}>
                    {CardContent}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
