"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type PlantForChat = {
  plant_id?: string;
  latin?: string;
  common?: string[];
  climate?: string;
  watering?: string;
  care_level?: string;
  ideallight?: string;
  toleratedlight?: string;
  growth_rate?: string;
  care_guidelines?: Record<string, string>;
  description?: {
    interesting_fact?: string;
    symbolism?: string;
    physical?: string;
  };
  image_url?: string;
};

type ChatMessage = { role: "user" | "plant"; content: string; audioSrc?: string };

const getPlantDisplayName = (plant: PlantForChat | null | undefined): string => {
  if (!plant) return "your plant";
  if (Array.isArray(plant.common) && plant.common.length > 0 && plant.common[0]) return plant.common[0];
  if (plant.latin && plant.latin.trim()) return plant.latin.trim();
  return "your plant";
};

export default function GardenChatPage() {
  const params = useSearchParams();
  const plantId = params.get("plant_id") || "";
  const username = params.get("username") || "";

  const [plant, setPlant] = useState<PlantForChat | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loadingPlant, setLoadingPlant] = useState(true);
  const [loadingReply, setLoadingReply] = useState(false);
  const [ttsLoadingIndex, setTtsLoadingIndex] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loadingReply]);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!plantId) {
        setError("We couldn't load this plant. Please return to your garden.");
        setLoadingPlant(false);
        return;
      }
      setLoadingPlant(true);
      setError(null);
      try {
        const res = await fetch(`${API_URL}/api/plant/${encodeURIComponent(plantId)}`);
        if (!res.ok) throw new Error(res.statusText || "Plant not found");
        const data = await res.json();
        if (cancelled) return;
        setPlant(data);
        const helloName = getPlantDisplayName(data);
        setMessages([
          {
            role: "plant",
            content: `Hi${username ? ` ${username}` : ""}, I’m ${helloName}. Talk to me 🌿`,
          },
        ]);
      } catch {
        if (!cancelled) {
          setError("We couldn't load this plant. Please return to your garden.");
        }
      } finally {
        if (!cancelled) setLoadingPlant(false);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [plantId, username]);

  const send = async () => {
    const text = input.trim();
    if (!text || loadingReply || !plant) return;

    const nextHistory = [...messages, { role: "user" as const, content: text }];
    setMessages(nextHistory);
    setInput("");
    setLoadingReply(true);

    try {
      const res = await fetch(`${API_URL}/api/garden/plant-chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          plant,
          userMessage: text,
          history: nextHistory,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Plant chat failed");
      setMessages((prev) => [
        ...prev,
        {
          role: "plant",
          content: data?.reply || "I'm feeling a little quiet today 🌿",
          audioSrc:
            typeof data?.audio_base64 === "string" && data.audio_base64.trim()
              ? `data:audio/mpeg;base64,${data.audio_base64}`
              : undefined,
        },
      ]);
    } catch {
      setMessages((prev) => [...prev, { role: "plant", content: "I'm feeling a little quiet today 🌿" }]);
    } finally {
      setLoadingReply(false);
    }
  };

  const speakPlantMessage = async (index: number) => {
    const msg = messages[index];
    if (!msg || msg.role !== "plant" || !msg.content.trim() || ttsLoadingIndex !== null) return;

    setTtsLoadingIndex(index);
    try {
      const res = await fetch(`${API_URL}/api/garden/plant-tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: msg.content }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Plant speech failed");
      const audioSrc =
        typeof data?.audio_base64 === "string" && data.audio_base64.trim()
          ? `data:audio/mpeg;base64,${data.audio_base64}`
          : undefined;
      if (!audioSrc) return;
      setMessages((prev) =>
        prev.map((item, i) => (i === index ? { ...item, audioSrc } : item))
      );
    } catch {
      // Keep UI stable if TTS fails; user can retry with the button.
    } finally {
      setTtsLoadingIndex(null);
    }
  };

  const titleName = getPlantDisplayName(plant);

  return (
    <div className="flex min-h-screen flex-col bg-white dark:bg-zinc-900">
      <div className="mx-auto w-full max-w-3xl flex-1 px-4 py-6">
        <Link
          href="/garden"
          className="mb-4 inline-flex items-center text-sm text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-300"
        >
          ← Back to Garden
        </Link>

        {error ? (
          <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800/50 dark:text-zinc-300">
            {error}
          </div>
        ) : (
          <>
            <div className="mb-4 overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800/50">
              <div className="flex gap-4 p-4">
                <div className="h-20 w-20 shrink-0 overflow-hidden rounded-lg bg-zinc-200 dark:bg-zinc-700">
                  {plant?.image_url ? (
                    <img src={plant.image_url} alt={titleName} className="h-full w-full object-cover" />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center text-xs text-zinc-400">
                      No image
                    </div>
                  )}
                </div>
                <div className="min-w-0">
                  <h1 className="text-xl font-semibold text-emerald-700 dark:text-emerald-400">{titleName}</h1>
                  {plant?.latin && (
                    <p className="text-sm text-zinc-600 dark:text-zinc-400">{plant.latin}</p>
                  )}
                </div>
              </div>
            </div>

            <div className="space-y-3">
              {loadingPlant && (
                <p className="text-sm text-zinc-500 dark:text-zinc-400">Loading plant...</p>
              )}
              {messages.map((m, i) => (
                <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-[15px] leading-relaxed ${
                      m.role === "user"
                        ? "bg-emerald-600 text-white"
                        : "bg-zinc-100 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200"
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{m.content}</p>
                    {m.role === "plant" && (
                      <div className="mt-2">
                        <button
                          type="button"
                          onClick={() => speakPlantMessage(i)}
                          disabled={ttsLoadingIndex !== null}
                          className="rounded-md border border-emerald-500 px-2 py-1 text-xs text-emerald-600 hover:bg-emerald-50 disabled:opacity-50 dark:text-emerald-300 dark:hover:bg-zinc-700"
                        >
                          {ttsLoadingIndex === i ? "Generating voice..." : "Speak"}
                        </button>
                        {m.audioSrc && (
                          <audio controls className="mt-2 w-full">
                            <source src={m.audioSrc} type="audio/mpeg" />
                          </audio>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loadingReply && (
                <div className="flex justify-start">
                  <div className="rounded-2xl bg-zinc-100 px-4 py-3 text-sm text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                    Plant is thinking...
                  </div>
                </div>
              )}
              <div ref={endRef} />
            </div>
          </>
        )}
      </div>

      <div className="shrink-0 border-t border-zinc-200 bg-white px-4 py-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2 rounded-2xl border border-zinc-200 bg-white px-4 py-3 shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
            <textarea
              placeholder={`Message ${titleName}...`}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={1}
              className="max-h-32 flex-1 resize-none bg-transparent text-[15px] text-zinc-800 placeholder:text-zinc-400 focus:outline-none dark:text-zinc-200"
              disabled={!!error || loadingPlant || loadingReply || !plant}
            />
            <button
              onClick={send}
              disabled={!!error || loadingPlant || loadingReply || !input.trim() || !plant}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-40 disabled:hover:bg-emerald-600"
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}


