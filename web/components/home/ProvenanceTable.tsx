import { PROVENANCE, type DataStatus } from "@/lib/provenance";

const STATUS_STYLE: Record<DataStatus, string> = {
  real: "bg-green-100 text-green-800 border-green-300",
  illustrative: "bg-amber-50 text-amber-800 border-amber-300",
};

const STATUS_LABEL: Record<DataStatus, string> = {
  real: "Real",
  illustrative: "Illustrative",
};

export function ProvenanceTable() {
  return (
    <div className="border border-border-formal bg-surface rounded-md shadow-flat overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-subtle text-primary-navy">
              {["Part of the dashboard", "Data source", "Status", "Notes"].map((h) => (
                <th key={h} scope="col" className="text-left font-semibold px-4 py-2 border-b border-border-formal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {PROVENANCE.map((row, i) => (
              <tr
                key={row.area}
                className={`${i % 2 === 1 ? "bg-slate-50/50" : ""} border-b border-border-formal last:border-b-0 align-top`}
              >
                <td className="px-4 py-2.5 font-medium text-text-primary">{row.area}</td>
                <td className="px-4 py-2.5 text-text-secondary">{row.source}</td>
                <td className="px-4 py-2.5">
                  <span className={`inline-block rounded-sm border px-2 py-0.5 text-[11px] font-semibold whitespace-nowrap ${STATUS_STYLE[row.status]}`}>
                    {STATUS_LABEL[row.status]}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-text-secondary">{row.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
