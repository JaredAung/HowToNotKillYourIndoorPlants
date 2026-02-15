"use client";

import Link from "next/link";
import { useState } from "react";
import { PlantCardMenu } from "../components/PlantCardMenu";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type GardenPlant = { latin: string; name: string; plant_id: string; image_url: string };

const PROFILE_LABELS: Record<string, string> = {
  experience_level: "Experience",
  climate: "Climate",
  light_availability: "Light",
  room_size: "Room size",
  max_plant_size_preference: "Max size",
  time_to_commit: "Time to commit",
  average_room_temp: "Room temp",
  average_sunlight_time: "Sun hours",
  use: "Use",
  symbolism: "Motivation",
  watering_preferences: "Watering",
  humidity_level: "Humidity",
  address: "Address",
};

const PROFILE_ORDER = [
  "experience_level",
  "climate",
  "light_availability",
  "room_size",
  "max_plant_size_preference",
  "address",
  "time_to_commit",
  "average_room_temp",
  "average_sunlight_time",
  "symbolism",
  "use",
  "watering_preferences",
  "humidity_level",
];

function formatProfileValue(val: unknown): string {
  if (val === true) return "Yes";
  if (val === false) return "No";
  if (val == null) return "—";
  if (Array.isArray(val)) return val.join(", ");
  return String(val).trim();
}

export default function GardenPage() {
  const [username, setUsername] = useState("");
  const [inputUsername, setInputUsername] = useState("");
  const [plants, setPlants] = useState<GardenPlant[]>([]);
  const [profile, setProfile] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchGarden = async (name: string) => {
    if (!name.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const [gardenRes, profileRes] = await Promise.all([
        fetch(`${API_URL}/api/garden?username=${encodeURIComponent(name.trim())}`),
        fetch(`${API_URL}/api/profile?username=${encodeURIComponent(name.trim())}`),
      ]);
      if (!gardenRes.ok) throw new Error(gardenRes.statusText);
      const gardenData = await gardenRes.json();
      setPlants(gardenData.plants ?? []);
      setUsername(name.trim());
      if (profileRes.ok) {
        const profileData = await profileRes.json();
        setProfile((profileData.profile ?? {}) as Record<string, unknown>);
      } else {
        setProfile({});
      }
    } catch {
      setError("Could not load your garden. Is the backend running?");
      setPlants([]);
      setProfile({});
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchGarden(inputUsername);
  };

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
              <p className="mt-4 text-sm text-red-600 dark:text-red-400">{error}</p>
            )}
          </div>
        </div>
      </div>
    );
  }

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
                  setProfile({});
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
            <p className="mb-4 text-sm text-red-600 dark:text-red-400">{error}</p>
          )}

          <div className="mb-6 rounded-xl border border-zinc-200 bg-zinc-50 px-4 py-3 dark:border-zinc-700 dark:bg-zinc-800/50">
            <p className="mb-2 text-xs font-medium text-zinc-500 dark:text-zinc-400">
              {username}&apos;s profile
            </p>
            {Object.keys(profile).length > 0 ? (
              <div className="grid gap-x-6 gap-y-2 text-sm text-zinc-700 dark:text-zinc-300 sm:grid-cols-2 lg:grid-cols-3">
                {[...PROFILE_ORDER, ...Object.keys(profile).filter((k) => !PROFILE_ORDER.includes(k) && k !== "hard_filter")]
                  .filter((key) => key in profile)
                  .map((key) => {
                    const val = profile[key];
                    const label = PROFILE_LABELS[key] ?? key.replace(/_/g, " ");
                    const display = formatProfileValue(val);
                    return (
                      <div key={key} className="flex gap-2">
                        <span className="shrink-0 text-zinc-500 dark:text-zinc-400">
                          {label}:
                        </span>
                        <span>{display}</span>
                      </div>
                    );
                  })}
              </div>
            ) : (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No profile found. Complete the Recommender flow to build your profile.
              </p>
            )}
          </div>

          {plants.length === 0 && !loading ? (
            <p className="text-zinc-600 dark:text-zinc-400">
              No plants in your garden yet. Add plants from the Recommender by
              saying &quot;I&apos;ll take the first one&quot; or &quot;add the
              croton to my garden&quot;.
            </p>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {plants.map((plant, i) => {
                const handleRemove = () => {
                  if (!plant.plant_id) return;
                  setPlants((p) => p.filter((x) => x.plant_id !== plant.plant_id));
                };
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
                        <div className="flex h-full w-full items-center justify-center text-xs text-zinc-400">
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
                  "block overflow-hidden rounded-xl border border-zinc-200 bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800/50";
                return (
                  <div key={plant.plant_id || i} className="relative">
                    {plant.plant_id && (
                      <PlantCardMenu
                        plantId={plant.plant_id}
                        plantName={plant.latin}
                        username={username}
                        onRemove={handleRemove}
                      />
                    )}
                    {plant.plant_id ? (
                      <Link href={`/plant/${plant.plant_id}`} className={cardClass}>
                        {CardContent}
                      </Link>
                    ) : (
                      <div className={cardClass}>{CardContent}</div>
                    )}
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
