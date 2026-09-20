import { ArrowRight, ExternalLink } from "lucide-react";
import Link from "next/link";
import { REPO_URL, SITE_NAME, repoFile } from "@/lib/site";
import { tabHref } from "@/lib/tabs";
import { ProvenanceTable } from "./ProvenanceTable";

const METHOD_CHIP = {
  xbrl: "bg-blue-50 text-secondary-blue border-blue-200",
  heuristic: "bg-slate-100 text-slate-700 border-slate-300",
  llm: "bg-amber-50 text-amber-800 border-amber-300",
} as const;

const PIPELINE = [
  {
    title: "Fetch",
    body: "Filings and their machine-tagged XBRL facts are downloaded from SEC EDGAR, within the SEC's fair-access rate limit.",
  },
  {
    title: "Split",
    body: "Each filing is divided into its sections (management's discussion, risk factors, and so on) so the model reads focused passages.",
  },
  {
    title: "Extract",
    body: "Tagged facts are taken as they are. Simple rules and the fine-tuned model then read the prose the tags don't cover.",
  },
  {
    title: "Repair and validate",
    body: "A five-stage parser recovers a result from messy model output, and a schema check confirms the fields are well-formed.",
  },
  {
    title: "Store",
    body: "Results go into PostgreSQL with the origin, confidence and model version of every value; repeat requests are served from a cache.",
  },
  {
    title: "Serve",
    body: "A FastAPI service exposes single and batch extraction, plus health and monitoring endpoints.",
  },
];

const RULES = [
  {
    title: "Tagged values always win",
    body: "If a figure is machine-tagged in the filing, that value is used. The model can fill gaps but can never silently overwrite a tagged fact.",
  },
  {
    title: "Every value says where it came from",
    body: null,
  },
  {
    title: "Mistakes are meant to be catchable",
    body: "Because origin and confidence are recorded, a person can check only the model-derived values instead of everything.",
  },
];

const LINKS = [
  { label: "README", href: repoFile("README.md"), note: "Setup, architecture and the evidence table" },
  { label: "Evidence and benchmarks", href: `${REPO_URL}#evidence-and-benchmarks`, note: "Which numbers are measured and which are still targets" },
  { label: "Model card", href: repoFile("MODEL_CARD.md"), note: "What the model is, how it was trained, and limits on its use" },
  { label: "Tagged-versus-model rule", href: repoFile("docs/BOUNDARY.md"), note: "Exactly how tagged facts and model output are merged" },
];

const h2 = "text-lg font-semibold text-primary-navy tracking-tight";
const body = "text-sm text-text-secondary leading-relaxed max-w-3xl";

export function HomeContent() {
  return (
    <div>
      <section className="bg-primary-navy text-white">
        <div className="max-w-5xl mx-auto px-6 py-12">
          <p className="text-[11px] uppercase tracking-widest text-white/60 font-mono">SEC EDGAR &middot; filings to structured data</p>
          <h1 className="mt-3 text-3xl md:text-4xl font-semibold tracking-tight">{SITE_NAME}</h1>
          <p className="mt-4 max-w-2xl text-base text-white/80 leading-relaxed">
            Reads SEC filings and turns the parts that no tag covers into structured data, recording where every
            value came from so a person can check it.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              href={tabHref("regulatory-filings")}
              className="inline-flex items-center gap-2 h-9 px-4 bg-white text-primary-navy text-xs font-semibold hover:bg-white/90 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
            >
              Open the dashboard
              <ArrowRight className="h-3.5 w-3.5" aria-hidden />
            </Link>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 h-9 px-4 border border-white/40 text-white text-xs font-semibold hover:bg-white/10 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
            >
              View on GitHub
              <ExternalLink className="h-3.5 w-3.5" aria-hidden />
              <span className="sr-only">(opens in a new tab)</span>
            </a>
          </div>
        </div>
      </section>

      <div className="max-w-5xl mx-auto px-6 py-10 flex flex-col gap-12">
        <section aria-labelledby="problem">
          <h2 id="problem" className={h2}>The problem</h2>
          <p className={`${body} mt-3`}>
            SEC filings are only partly structured. Core statement figures such as revenue, net income and balance-sheet
            lines are usually machine-tagged (iXBRL). But much of what a filing discloses, including management&apos;s
            discussion, risk factors, footnotes and non-GAAP reconciliations, is free-form prose that no tag covers.
            Rule-based parsers break down on that prose.
          </p>
        </section>

        <section aria-labelledby="model" className="border-t border-border-formal pt-8">
          <h2 id="model" className={h2}>What the fine-tuned model does</h2>
          <div className="mt-3 grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="flex flex-col gap-3">
              <p className={body}>
                Think of a filing as a long letter with a few spreadsheets attached. The tagged numbers are the
                spreadsheets. <strong className="text-text-primary">The model is the reader of the letter.</strong>{" "}
                Given a passage of filing text, it fills out a fixed form (company, form type, dates, revenue, net
                income, total assets, earnings per share and so on) and reports how confident it is.
              </p>
              <p className={body}>
                It starts as a general-purpose open model (Llama 3.1 8B) and is fine-tuned, meaning given extra
                practice, on exactly this task using QLoRA, a low-cost method that trains a small add-on rather than
                the whole model.
              </p>
            </div>

            <ul className="border border-border-formal bg-surface divide-y divide-border-formal">
              {RULES.map((rule) => (
                <li key={rule.title} className="px-4 py-3">
                  <p className="text-sm font-semibold text-text-primary">{rule.title}</p>
                  {rule.body ? (
                    <p className="text-xs text-text-secondary leading-relaxed mt-1">{rule.body}</p>
                  ) : (
                    <p className="text-xs text-text-secondary leading-relaxed mt-1">
                      Each value is labeled{" "}
                      <span className={`inline-block rounded-sm border px-1.5 py-px font-mono text-[11px] ${METHOD_CHIP.xbrl}`}>xbrl</span>{" "}
                      (read from a tag),{" "}
                      <span className={`inline-block rounded-sm border px-1.5 py-px font-mono text-[11px] ${METHOD_CHIP.heuristic}`}>heuristic</span>{" "}
                      (a simple rule) or{" "}
                      <span className={`inline-block rounded-sm border px-1.5 py-px font-mono text-[11px] ${METHOD_CHIP.llm}`}>llm</span>{" "}
                      (the model), with a confidence score.
                    </p>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section aria-labelledby="how" className="border-t border-border-formal pt-8">
          <h2 id="how" className={h2}>How it works</h2>
          <ol className="mt-4 border border-border-formal bg-surface divide-y divide-border-formal">
            {PIPELINE.map((step, i) => (
              <li key={step.title} className="grid grid-cols-[2.5rem_1fr] sm:grid-cols-[2.5rem_11rem_1fr] gap-x-3 gap-y-1 px-4 py-3 items-baseline">
                <span className="font-mono text-xs text-text-muted tabular-figures">{i + 1}</span>
                <span className="text-sm font-semibold text-text-primary">{step.title}</span>
                <span className="text-xs text-text-secondary leading-relaxed col-span-2 sm:col-span-1 col-start-2 sm:col-start-auto">
                  {step.body}
                </span>
              </li>
            ))}
          </ol>
        </section>

        <section id="data-status" aria-labelledby="dashboard" className="border-t border-border-formal pt-8 scroll-mt-4">
          <h2 id="dashboard" className={h2}>What the dashboard shows, and what it doesn&apos;t</h2>
          <p className={`${body} mt-3 mb-4`}>
            <strong className="text-text-primary">The dashboard tabs do not run the model.</strong> They present filing
            listings and fund data from SEC sources that are already structured; the model is a separate extraction
            engine behind the API. The dashboard is also being moved from placeholder figures to real SEC data one part
            at a time, so this table says exactly which parts are real today.
          </p>
          <ProvenanceTable />
        </section>

        <section aria-labelledby="status" className="border-t border-border-formal pt-8">
          <h2 id="status" className={h2}>Where things stand</h2>
          <p className={`${body} mt-3`}>
            The extraction pipeline, parser, storage layer and API are built and covered by automated tests. No
            finished fine-tuning run has been published yet, so any accuracy figure in the docs is a target rather than
            a result; the evidence table in the README lists which numbers are measured. This page deliberately quotes
            no metrics so it can&apos;t drift out of date.
          </p>
        </section>

        <section aria-labelledby="read" className="border-t border-border-formal pt-8">
          <h2 id="read" className={h2}>Read more</h2>
          <ul className="mt-4 border border-border-formal bg-surface divide-y divide-border-formal">
            {LINKS.map((link) => (
              <li key={link.label}>
                <a
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-0.5 px-4 py-3 hover:bg-subtle transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-secondary-blue"
                >
                  <span className="text-sm font-semibold text-secondary-blue inline-flex items-center gap-1.5">
                    {link.label}
                    <ExternalLink className="h-3 w-3" aria-hidden />
                    <span className="sr-only">(opens in a new tab)</span>
                  </span>
                  <span className="text-xs text-text-muted">{link.note}</span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
