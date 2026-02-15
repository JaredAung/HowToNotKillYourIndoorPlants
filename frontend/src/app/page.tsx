"use client";

import { useState, useRef, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type PlantRecommendation = { name: string; image_url: string; explanation: string; plant_id?: string };
type Message = {
  role: "user" | "assistant";
  text: string;
  recommendations?: PlantRecommendation[];
};

const INITIAL_MESSAGE: Message = {
  role: "assistant",
  text: "Hi! What's your name?",
};

const normalizeName = (raw: unknown): string => {
  const text = String(raw ?? "").trim();
  if (!text) return "Plant recommendation";
  const lines = text
    .split(/\r?\n+/)
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length <= 1) return text;
  return lines[0];
};

const normalizeExplanation = (raw: unknown, name: string): string => {
  const text = String(raw ?? "").trim();
  if (!text) return "";
  // If the whole explanation is just the plant name, don't show it
  if (text.toLowerCase() === name.toLowerCase()) return "";
  const lines = text
    .split(/\r?\n+/)
    .map((l) => l.trim())
    .filter(Boolean);
  const cleaned = lines.filter((line) => line.toLowerCase() !== name.toLowerCase()).join("\n\n").trim();
  return cleaned;
};

const normalizeRecommendations = (raw: unknown): PlantRecommendation[] => {
  if (!Array.isArray(raw)) return [];
  return raw.map((item) => {
    const rec = (item ?? {}) as Partial<PlantRecommendation>;
    const name = normalizeName(rec.name);
    return {
      name,
      image_url: typeof rec.image_url === "string" ? rec.image_url : "",
      explanation: normalizeExplanation(rec.explanation, name),
    };
  });
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [progressLabel, setProgressLabel] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => scrollToBottom(), [messages, loading]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((m) => [...m, { role: "user", text }]);
    setLoading(true);
    setProgressLabel(null);

    try {
      const res = await fetch(`${API_URL}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      if (!res.ok) throw new Error(res.statusText);

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      if (!reader) throw new Error("No response body");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const payload = JSON.parse(line.slice(6));
              if (payload.type === "progress") {
                setProgressLabel(payload.label ?? "Working...");
              } else if (payload.type === "done") {
                const data = payload.data;
                setSessionId(data.session_id);
                const recs = normalizeRecommendations(data.recommendations ?? []);
                const greeting =
                  typeof data.greeting === "string" ? data.greeting.trim() : "";
                const responseText =
                  typeof data.response === "string" ? data.response : "";
                setMessages((m) => [
                  ...m,
                  {
                    role: "assistant",
                    text: greeting || responseText,
                    recommendations: recs.length > 0 ? recs : undefined,
                  },
                ]);
              } else if (payload.type === "error") {
                setMessages((m) => [
                  ...m,
                  {
                    role: "assistant",
                    text: `Error: ${payload.message ?? "Something went wrong"}`,
                  },
                ]);
              }
            } catch {
              // ignore parse errors for partial chunks
            }
          }
        }
      }
    } catch {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: "Could not connect to the server. Is the backend running?",
        },
      ]);
    } finally {
      setLoading(false);
      setProgressLabel(null);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-white dark:bg-zinc-900">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 py-6">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`group flex w-full gap-4 py-4 ${
                msg.role === "user"
                  ? "justify-end"
                  : "justify-start"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500 text-white">
                  <svg
                    className="h-4 w-4"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" />
                  </svg>
                </div>
              )}
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-emerald-600 text-white"
                    : "bg-zinc-100 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200"
                }`}
              >
                {msg.role === "assistant" && msg.recommendations?.length ? (
                  <div className="space-y-4">
                    {msg.text && (
                      <p className="whitespace-pre-wrap text-[15px] leading-relaxed">{msg.text}</p>
                    )}
                    {msg.recommendations.map((rec, j) => (
                      <div
                        key={j}
                        className="flex gap-4 items-start rounded-xl overflow-hidden bg-white/50 dark:bg-zinc-700/50 p-3"
                      >
                        <div className="shrink-0 w-24 h-24 rounded-lg overflow-hidden bg-zinc-200 dark:bg-zinc-600">
                          {rec.image_url ? (
                            <img
                              src={rec.image_url}
                              alt={rec.name}
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-zinc-400 text-xs">
                              No image
                            </div>
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-emerald-700 dark:text-emerald-400 mb-1">
                            {rec.name}
                          </p>
                          {rec.explanation ? (
                            <p className="text-[14px] leading-relaxed whitespace-pre-wrap">
                              {rec.explanation}
                            </p>
                          ) : null}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap text-[15px] leading-relaxed">{msg.text}</p>
                )}
              </div>
              {msg.role === "user" && (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-zinc-300 dark:bg-zinc-600">
                  <svg
                    className="h-4 w-4 text-zinc-600 dark:text-zinc-300"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                  </svg>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex w-full gap-4 py-4">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500 text-white">
                <svg
                  className="h-4 w-4"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" />
                </svg>
              </div>
              <div className="flex flex-col gap-1 rounded-2xl bg-zinc-100 px-4 py-3 dark:bg-zinc-800">
                <div className="flex items-center gap-1">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-400 [animation-delay:-0.3s]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-400 [animation-delay:-0.15s]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-400" />
                </div>
                {progressLabel && (
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    {progressLabel}
                  </p>
                )}
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area - ChatGPT style */}
      <div className="shrink-0 border-t border-zinc-200 bg-white px-4 py-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2 rounded-2xl border border-zinc-200 bg-white px-4 py-3 shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
            <textarea
              placeholder={messages.length === 1 ? "Enter your name..." : "Message How To Not Kill Your Indoor Plants..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              rows={1}
              className="max-h-32 flex-1 resize-none bg-transparent text-[15px] text-zinc-800 placeholder:text-zinc-400 focus:outline-none dark:text-zinc-200"
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-40 disabled:hover:bg-emerald-600"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </button>
          </div>
          <p className="mt-2 text-center text-xs text-zinc-500 dark:text-zinc-400">
            Plant care assistant can make mistakes. Consider checking important information.
          </p>
        </div>
      </div>
    </div>
  );
}
