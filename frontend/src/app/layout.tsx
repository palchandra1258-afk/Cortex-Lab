import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cortex Lab · AI Chat Interface",
  description:
    "Advanced chat interface powered by DeepSeek-R1-7B with reasoning visualization",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
