import type { Metadata } from "next";
import { headers } from "next/headers";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const title = "C-SKL Atlas · Biological Dataset Discovery";
const description =
  "A versioned evidence map for exploring molecular similarity, scientific-text concordance, gene drivers, and provenance across biological datasets.";

function parseHttpOrigin(value: string, source: string): URL {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${source} must be an absolute HTTP or HTTPS origin.`);
  }
  if (
    !["http:", "https:"].includes(parsed.protocol) ||
    parsed.username ||
    parsed.password ||
    parsed.pathname !== "/" ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error(`${source} must contain only an HTTP or HTTPS origin.`);
  }
  return new URL(parsed.origin);
}

export async function generateMetadata(): Promise<Metadata> {
  const configuredOrigin = process.env.CSKL_ATLAS_PUBLIC_ORIGIN?.trim();
  let metadataBase: URL;
  if (configuredOrigin) {
    metadataBase = parseHttpOrigin(configuredOrigin, "CSKL_ATLAS_PUBLIC_ORIGIN");
  } else {
    const requestHeaders = await headers();
    const host =
      requestHeaders.get("x-forwarded-host")?.split(",", 1)[0]?.trim() ??
      requestHeaders.get("host") ??
      "localhost:3000";
    const forwardedProtocol = requestHeaders
      .get("x-forwarded-proto")
      ?.split(",", 1)[0]
      ?.trim()
      .toLowerCase();
    const protocol = forwardedProtocol === "http" || forwardedProtocol === "https"
      ? forwardedProtocol
      : host.startsWith("localhost")
        ? "http"
        : "https";
    metadataBase = parseHttpOrigin(`${protocol}://${host}`, "request host");
  }
  const socialImage = new URL("/og.png", metadataBase).toString();

  return {
    metadataBase,
    title,
    description,
    openGraph: {
      title,
      description,
      type: "website",
      images: [
        {
          url: socialImage,
          width: 1744,
          height: 909,
          alt: "C-SKL Atlas evidence-map preview",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [socialImage],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
