import type { Metadata } from "next";
import { HomeContent } from "@/components/home/HomeContent";

export const metadata: Metadata = {
  description:
    "Turns the parts of SEC filings that no tag covers into structured data, and records where every value came from.",
};

export default function HomePage() {
  return <HomeContent />;
}
