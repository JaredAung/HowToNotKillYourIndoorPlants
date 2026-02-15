"use client";

import Link from "next/link";
import { useState } from "react";

type PlantCardMenuProps = {
  plantId: string;
  plantName: string;
  username?: string;
  onRemove?: () => void;
};

export function PlantCardMenu({
  plantId,
  plantName,
  username,
  onRemove,
}: PlantCardMenuProps) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className="absolute right-2 top-2 z-10"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/90 text-zinc-600 shadow-sm transition-colors hover:bg-white dark:bg-zinc-800/90 dark:text-zinc-300 dark:hover:bg-zinc-700"
        aria-label="Plant options"
      >
        <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
          <circle cx="12" cy="6" r="1.5" />
          <circle cx="12" cy="12" r="1.5" />
          <circle cx="12" cy="18" r="1.5" />
        </svg>
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 min-w-[140px] rounded-lg border border-zinc-200 bg-white py-1 shadow-lg dark:border-zinc-700 dark:bg-zinc-800">
          <Link
            href={`/?plant=${encodeURIComponent(plantId)}`}
            className="block px-3 py-2 text-sm text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-700"
            onClick={(e) => e.stopPropagation()}
          >
            Talk to plant
          </Link>
          {username && onRemove ? (
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                onRemove();
                setOpen(false);
              }}
              className="block w-full px-3 py-2 text-left text-sm text-red-600 hover:bg-zinc-100 dark:text-red-400 dark:hover:bg-zinc-700"
            >
              Death
            </button>
          ) : (
            <Link
              href="/garden"
              className="block px-3 py-2 text-sm text-red-600 hover:bg-zinc-100 dark:text-red-400 dark:hover:bg-zinc-700"
              onClick={(e) => e.stopPropagation()}
            >
              Death
            </Link>
          )}
        </div>
      )}
    </div>
  );
}
