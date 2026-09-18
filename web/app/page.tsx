import { redirect } from "next/navigation";
import { tabHref } from "@/lib/tabs";

export default function RootPage() {
  redirect(tabHref("portfolio-matrix"));
}
