"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type PlantDetail = {
  plant_id: string;
  latin: string;
  common: string[];
  climate: string;
  size_bucket: string;
  use: string[];
  care_level: string;
  ideallight: string;
  watering: string;
  description: { physical?: string };
  image_url: string;
};

export default function PlantDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const [plant, setPlant] = useState<PlantDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`${API_URL}/api/plant/${encodeURIComponent(id)}`)
      .then((res) => {
        if (!res.ok) throw new Error(res.status === 404 ? "Plant not found" : res.statusText);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setPlant(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message || "Failed to load plant");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (loading) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-white dark:bg-zinc-900">
        <p className="text-zinc-500">Loading...</p>
      </div>
    );
  }

  if (error || !plant) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-white dark:bg-zinc-900 px-4">
        <p className="mb-4 text-zinc-600 dark:text-zinc-400">
          {error || "Plant not found"}
        </p>
        <Link
          href="/garden"
          className="text-emerald-600 hover:text-emerald-700 dark:text-emerald-400"
        >
          ← Back to Garden
        </Link>
      </div>
    );
  }

  const commonNames = Array.isArray(plant.common) ? plant.common : [];
  const useList = Array.isArray(plant.use) ? plant.use : [];
  const physical = plant.description?.physical || "";

  return (
    <div className="min-h-screen bg-white dark:bg-zinc-900">
      <div className="mx-auto max-w-2xl px-4 py-8">
        <Link
          href="/garden"
          className="mb-6 inline-flex items-center text-sm text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-300"
        >
          ← Back to Garden
        </Link>

        <div className="overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800/50">
          {/* Image */}
          <div className="aspect-[4/3] w-full overflow-hidden bg-zinc-200 dark:bg-zinc-700">
            {plant.image_url ? (
              <img
                src={plant.image_url}
                alt={plant.latin}
                className="h-full w-full object-cover"
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center text-zinc-400">
                No image
              </div>
            )}
          </div>

          {/* Content */}
          <div className="p-6">
            <h1 className="text-2xl font-bold text-emerald-700 dark:text-emerald-400">
              {plant.latin}
            </h1>
            {commonNames.length > 0 && (
              <p className="mt-1 text-zinc-600 dark:text-zinc-400">
                {commonNames.join(", ")}
              </p>
            )}

            <dl className="mt-6 space-y-4">
              {plant.climate && (
                <div>
                  <dt className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                    Climate
                  </dt>
                  <dd className="mt-0.5 text-zinc-800 dark:text-zinc-200">
                    {plant.climate}
                  </dd>
                </div>
              )}
              {plant.size_bucket && (
                <div>
                  <dt className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                    Size
                  </dt>
                  <dd className="mt-0.5 text-zinc-800 dark:text-zinc-200">
                    {plant.size_bucket}
                  </dd>
                </div>
              )}
              {plant.care_level && (
                <div>
                  <dt className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                    Care level
                  </dt>
                  <dd className="mt-0.5 text-zinc-800 dark:text-zinc-200">
                    {plant.care_level}
                  </dd>
                </div>
              )}
              {plant.ideallight && (
                <div>
                  <dt className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                    Light
                  </dt>
                  <dd className="mt-0.5 text-zinc-800 dark:text-zinc-200">
                    {plant.ideallight}
                  </dd>
                </div>
              )}
              {plant.watering && (
                <div>
                  <dt className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                    Watering
                  </dt>
                  <dd className="mt-0.5 text-zinc-800 dark:text-zinc-200">
                    {plant.watering}
                  </dd>
                </div>
              )}
              {useList.length > 0 && (
                <div>
                  <dt className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                    Use
                  </dt>
                  <dd className="mt-0.5 text-zinc-800 dark:text-zinc-200">
                    {useList.join(", ")}
                  </dd>
                </div>
              )}
              {physical && (
                <div>
                  <dt className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                    Description
                  </dt>
                  <dd className="mt-0.5 text-zinc-800 dark:text-zinc-200 whitespace-pre-wrap">
                    {physical}
                  </dd>
                </div>
              )}
            </dl>
          </div>
        </div>
      </div>
    </div>
  );
}
