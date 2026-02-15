import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono, Playfair_Display } from "next/font/google";
import { Nav } from "./components/Nav";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const playfair = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
  weight: ["600", "700"],
});

export const metadata: Metadata = {
  title: "How To Not Kill Your Indoor Plants",
  description: "Indoor plant care assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${playfair.variable} antialiased`}
      >
        <header className="shrink-0 border-b border-zinc-200 px-4 py-4 dark:border-zinc-800">
          <div className="mx-auto flex max-w-4xl items-center justify-between gap-4">
            <Link href="/" className="shrink-0">
              <h1 className="text-2xl font-bold tracking-tight [font-family:var(--font-playfair)] sm:text-3xl">
                <span className="bg-gradient-to-r from-emerald-600 via-emerald-700 to-teal-700 bg-clip-text text-transparent dark:from-emerald-400 dark:via-emerald-500 dark:to-teal-500">
                  How To Not Kill Your Indoor Plants
                </span>
              </h1>
            </Link>
            <Nav />
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
