import type { FilingRow, FilingStatus } from "@/lib/types";

const STATUS_STYLE: Record<FilingStatus, string> = {
  accepted: "bg-green-100 text-green-800 border-green-300",
  in_review: "bg-slate-100 text-slate-800 border-slate-300",
  flagged: "bg-red-100 text-red-800 border-red-300",
};

const STATUS_LABEL: Record<FilingStatus, string> = {
  accepted: "Processed / Accepted",
  in_review: "In Review",
  flagged: "Filing Deficit / Flagged",
};

function StatusBadge({ status }: { status: FilingStatus }) {
  const weight = status === "in_review" ? "" : "font-semibold";
  return (
    <span className={`inline-block rounded-sm border px-2 py-0.5 text-[11px] ${weight} ${STATUS_STYLE[status]}`}>
      {STATUS_LABEL[status]}
    </span>
  );
}

const HEADERS = ["Filing Entity", "Form Type", "Filing Date", "Authorized Signatory", "Status Flag"];

export function FilingsLedger({ rows }: { rows: FilingRow[] }) {
  return (
    <div className="border border-border-formal bg-surface rounded-md shadow-flat overflow-hidden">
      <div className="px-4 py-3 border-b border-border-formal bg-subtle flex items-baseline justify-between gap-4">
        <h2 className="text-sm font-semibold text-primary-navy">Corporate Ledger &amp; Filings Stream</h2>
        <span className="text-[11px] text-text-muted font-mono tabular-figures" aria-live="polite">
          {rows.length} {rows.length === 1 ? "filing" : "filings"}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-subtle text-primary-navy">
              {HEADERS.map((h) => (
                <th key={h} scope="col" className="text-left font-semibold px-4 py-2 border-b border-border-formal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={HEADERS.length} className="px-4 py-10 text-center text-text-muted">
                  No filings match the current filters.
                </td>
              </tr>
            ) : (
              rows.map((row, i) => (
                <tr
                  key={`${row.cik}-${row.filingDate}-${row.formType}`}
                  className={`${i % 2 === 1 ? "bg-slate-50/50" : ""} border-b border-border-formal last:border-b-0`}
                >
                  <td className="px-4 py-2.5">
                    <div className="text-text-primary font-medium">{row.entity}</div>
                    <div className="text-text-muted font-mono text-[11px]">
                      {row.ticker} &middot; CIK {row.cik} &middot; {row.sector}
                    </div>
                  </td>
                  <td className="px-4 py-2.5 font-mono text-text-secondary whitespace-nowrap">{row.formType}</td>
                  <td className="px-4 py-2.5 tabular-figures text-text-secondary whitespace-nowrap">{row.filingDate}</td>
                  <td className="px-4 py-2.5 text-text-secondary">{row.signatory}</td>
                  <td className="px-4 py-2.5">
                    <StatusBadge status={row.status} />
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
