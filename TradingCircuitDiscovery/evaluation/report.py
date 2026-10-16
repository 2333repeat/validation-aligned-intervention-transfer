from __future__ import annotations

import json
from pathlib import Path


def generate_report(metrics: dict, circuits: dict, output_dir: str | Path, title: str = "Trading Circuit Discovery Report") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "circuits.json").write_text(json.dumps(circuits, indent=2), encoding="utf-8")
    lines = [f"# {title}", "", "## Metrics", ""]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Circuits", ""])
    for name, neurons in circuits.items():
        lines.append(f"- `{name}`: {', '.join(neurons)}")
    report_path = out / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    generate_pdf_report(lines, out / "report.pdf")
    return report_path


def generate_pdf_report(lines: list[str], output_path: str | Path) -> Path:
    path = Path(output_path)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.95, "\n".join(lines[:45]), va="top", family="monospace", fontsize=9)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    except Exception:
        minimal_pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
        path.write_bytes(minimal_pdf)
    return path
