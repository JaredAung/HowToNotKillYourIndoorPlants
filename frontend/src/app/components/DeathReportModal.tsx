"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CAUSES = [
  "underwatering",
  "overwatering",
  "light",
  "pests",
  "temperature",
  "humidity",
  "soil/drainage",
  "nutrition",
  "unknown",
] as const;

type DeathReportModalProps = {
  plantId: string;
  plantName: string;
  username?: string;
  onClose: () => void;
  onComplete?: () => void;
};

export function DeathReportModal({
  plantId,
  plantName,
  username,
  onClose,
  onComplete,
}: DeathReportModalProps) {
  const [step, setStep] = useState<"form" | "confirm" | "prevention">("form");
  const [cause, setCause] = useState<string[]>([]);
  const [details, setDetails] = useState("");
  const [reminderSystem, setReminderSystem] = useState(false);
  const [wherePlaced, setWherePlaced] = useState("");
  const [lastWateredDate, setLastWateredDate] = useState("");
  const [travelAway, setTravelAway] = useState(false);
  const [preventionOptions, setPreventionOptions] = useState<string[]>([]);
  const [selectedPreventions, setSelectedPreventions] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleCause = (c: string) => {
    setCause((prev) =>
      prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]
    );
  };

  const handleSubmitForm = async () => {
    if (cause.length === 0) {
      setError("Please select at least one cause");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/death-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: username || "",
          plant_id: plantId,
          cause,
          details: details.trim(),
          reminder_system: reminderSystem,
          where_placed: wherePlaced.trim(),
          last_watered_date: lastWateredDate.trim() || null,
          travel_away: travelAway,
        }),
      });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      setPreventionOptions(data.prevention_options ?? []);
      setStep("prevention");
    } catch {
      setError("Failed to submit report");
    } finally {
      setLoading(false);
    }
  };

  const togglePrevention = (opt: string) => {
    setSelectedPreventions((prev) => {
      const next = new Set(prev);
      if (next.has(opt)) next.delete(opt);
      else next.add(opt);
      return next;
    });
  };

  const handleFinish = async () => {
    setLoading(true);
    setError(null);
    try {
      // Apply prevention actions and remove plant from garden
      if (username) {
        await fetch(`${API_URL}/api/death-report/apply-preventions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            username,
            selected_preventions: Array.from(selectedPreventions),
            plant_id: plantId,
          }),
        });
        if (onComplete) onComplete();
      }
    } catch {
      setError("Failed to complete actions");
    } finally {
      setLoading(false);
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-2xl border border-zinc-200 bg-white shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
        <div className="sticky top-0 flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-700">
          <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">
            💀 Death Report
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
            aria-label="Close"
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-4 space-y-4">
          {error && (
            <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400">
              {error}
            </p>
          )}

          {step === "form" && (
            <>
              <div>
                <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                  Plant
                </label>
                <p className="mt-0.5 text-zinc-600 dark:text-zinc-400">{plantName}</p>
                <input type="hidden" value={plantId} readOnly />
              </div>

              <div>
                <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                  Date
                </label>
                <p className="mt-0.5 text-zinc-600 dark:text-zinc-400">
                  {new Date().toLocaleDateString()}
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
                  Cause (select all that apply)
                </label>
                <div className="flex flex-wrap gap-2">
                  {CAUSES.map((c) => (
                    <label
                      key={c}
                      className={`inline-flex cursor-pointer items-center rounded-lg border px-3 py-1.5 text-sm transition-colors ${
                        cause.includes(c)
                          ? "border-emerald-500 bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400"
                          : "border-zinc-200 bg-zinc-50 text-zinc-600 hover:border-zinc-300 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-400"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={cause.includes(c)}
                        onChange={() => toggleCause(c)}
                        className="sr-only"
                      />
                      {c.replace("/", " / ")}
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label htmlFor="details" className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                  Details (optional)
                </label>
                <textarea
                  id="details"
                  value={details}
                  onChange={(e) => setDetails(e.target.value)}
                  placeholder="e.g. forgot to water for 2 days"
                  rows={2}
                  className="mt-1 w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-zinc-800 placeholder:text-zinc-400 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-200"
                />
              </div>

              <div className="border-t border-zinc-200 pt-4 dark:border-zinc-700">
                <p className="mb-3 text-sm font-medium text-zinc-700 dark:text-zinc-300">
                  Care context
                </p>
                <div className="space-y-3">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={reminderSystem}
                      onChange={(e) => setReminderSystem(e.target.checked)}
                      className="rounded border-zinc-300 text-emerald-600 focus:ring-emerald-500"
                    />
                    <span className="text-sm text-zinc-600 dark:text-zinc-400">
                      Reminder system on?
                    </span>
                  </label>
                  <div>
                    <label htmlFor="where_placed" className="block text-xs text-zinc-500 dark:text-zinc-400">
                      Where placed (light level)
                    </label>
                    <input
                      id="where_placed"
                      type="text"
                      value={wherePlaced}
                      onChange={(e) => setWherePlaced(e.target.value)}
                      placeholder="e.g. north window, low light"
                      className="mt-0.5 w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-800 placeholder:text-zinc-400 focus:border-emerald-500 focus:outline-none dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-200"
                    />
                  </div>
                  <div>
                    <label htmlFor="last_watered" className="block text-xs text-zinc-500 dark:text-zinc-400">
                      Last watered date (optional)
                    </label>
                    <input
                      id="last_watered"
                      type="date"
                      value={lastWateredDate}
                      onChange={(e) => setLastWateredDate(e.target.value)}
                      className="mt-0.5 w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-800 focus:border-emerald-500 focus:outline-none dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-200"
                    />
                  </div>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={travelAway}
                      onChange={(e) => setTravelAway(e.target.checked)}
                      className="rounded border-zinc-300 text-emerald-600 focus:ring-emerald-500"
                    />
                    <span className="text-sm text-zinc-600 dark:text-zinc-400">
                      Travel / away recently?
                    </span>
                  </label>
                </div>
              </div>

              <div className="flex gap-2 pt-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="flex-1 rounded-lg border border-zinc-200 px-4 py-2 text-sm font-medium text-zinc-700 hover:bg-zinc-50 dark:border-zinc-600 dark:text-zinc-300 dark:hover:bg-zinc-800"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSubmitForm}
                  disabled={loading || cause.length === 0}
                  className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50"
                >
                  {loading ? "Submitting..." : "Submit report"}
                </button>
              </div>
            </>
          )}

          {step === "prevention" && (
            <>
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                Based on your report, here are prevention options. Select what you&apos;d like to change:
              </p>
              <div className="space-y-2">
                {preventionOptions.map((opt) => (
                  <label
                    key={opt}
                    className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2 text-sm transition-colors ${
                      selectedPreventions.has(opt)
                        ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20"
                        : "border-zinc-200 dark:border-zinc-600"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedPreventions.has(opt)}
                      onChange={() => togglePrevention(opt)}
                      className="rounded border-zinc-300 text-emerald-600 focus:ring-emerald-500"
                    />
                    {opt}
                  </label>
                ))}
              </div>
              <div className="flex gap-2 pt-2">
                <button
                  type="button"
                  onClick={handleFinish}
                  disabled={loading}
                  className="w-full rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                >
                  {loading
                    ? "Removing..."
                    : username
                      ? "Done — remove from garden"
                      : "Done"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
