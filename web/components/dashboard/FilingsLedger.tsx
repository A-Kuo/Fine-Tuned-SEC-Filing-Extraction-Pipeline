import { filingsLedger, type FilingStatus } from "@/lib/mock-data";

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
    <span
      className={`inline-block rounded-sm border px-2 py-0.5 text-[11px] ${weight} ${STATUS_STYLE[status]}`}
    >
      {STATUS_LABEL[status]}
    </span>
  );
}

export function FilingsLedger() {
  return (
    <div className="border border-border-formal bg-surface rounded-md shadow-flat overflow-hidden">
      <div className="px-4 py-3 border-b border-border-formal bg-subtle">
        <h2 className="text-sm font-semibold text-primary-navy">Corporate Ledger &amp; Filings Stream</h2>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-subtle text-primary-navy">
              <th className="text-left font-semibold px-4 py-2 border-b border-border-formal">Filing Entity</th>
              <th className="text-left font-semibold px-4 py-2 border-b border-border-formal">Form Type</th>
              <th className="text-left font-semibold px-4 py-2 border-b border-border-formal">Filing Date</th>
              <th className="text-left font-semibold px-4 py-2 border-b border-border-formal">Authorized Signatory</th>
              <th className="text-left font-semibold px-4 py-2 border-b border-border-formal">Status Flag</th>
            </tr>
          </thead>
          <tbody>
            {filingsLedger.map((row, i) => (
              <tr
                key={row.cik + row.filingDate}
                className={`${i % 2 === 1 ? "bg-slate-50/50" : ""} border-b border-border-formal last:border-b-0`}
              >
                <td className="px-4 py-2.5">
                  <div className="text-text-primary font-medium">{row.entity}</div>
                  <div className="text-text-muted font-mono text-[11px]">
                    {row.ticker} &middot; CIK {row.cik}
                  </div>
                </td>
                <td className="px-4 py-2.5 font-mono text-text-secondary">{row.formType}</td>
                <td className="px-4 py-2.5 tabular-figures text-text-secondary">{row.filingDate}</td>
                <td className="px-4 py-2.5 text-text-secondary">{row.signatory}</td>
                <td className="px-4 py-2.5">
                  <StatusBadge status={row.status} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
