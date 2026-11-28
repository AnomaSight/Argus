import type { Cell, ResultTable } from "../api/types";

const show = (c: Cell) => (c === null ? "" : typeof c === "number" ? +c.toFixed(4) : c);

export default function ResultsTable({ table }: { table: ResultTable }) {
  const numeric = table.columns.map((_, i) => table.rows.some((r) => typeof r[i] === "number"));
  return (
    <section className="table-block">
      <h2>{table.name}</h2>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {table.columns.map((c, i) => (
                <th key={c} className={numeric[i] ? "num" : undefined}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, r) => (
              <tr key={r}>
                {row.map((cell, i) => (
                  <td key={i} className={numeric[i] ? "num" : undefined}>
                    {show(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
