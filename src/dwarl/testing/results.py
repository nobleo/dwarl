"""One compact report for either simulator or a paired workload."""


def write_report(directory, workload, rows, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    from dwarl.datasets.generation.mapdata import MapData

    report = directory / "report"
    report.mkdir(exist_ok=True)
    names = list(rows)
    lines = ["# Navigation test", "", f"Checkpoint: `{summary['checkpoint']}`", "",
             f"Workload: `{summary['workload']}`", "", f"Status: **{summary['status']}**", "",
             "Both backends share case geometry, start poses and goals. Nav2 computes its own global plan.",
             "Unavailable measurements remain null; ground-truth collision sampling differs between simulators.", "",
             "| Case | Simulator | Status | Success | Collision | Distance (m) |", "|---|---|---|---|---|---|"]
    for case in workload["cases"]:
        selected = {name: next((r for r in rows[name] if r["case_id"] == case["id"]), None) for name in names}
        for name, row in selected.items():
            if row:
                lines.append(f"| {case['name']} / {case['repeat']} | {name} | {row['status']} | "
                             f"{row.get('success', 'unavailable')} | {row.get('collision', 'unavailable')} | "
                             f"{row.get('travelled_distance', 'unavailable')} |")
        if not any(row and row.get("trajectory") for row in selected.values()):
            continue
        md = MapData.load(directory / case["map_file"])
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
        for ax, name in zip(axes[0], names, strict=True):
            import math

            for kind, x, y, angle, hx, hy in md.primitives:
                if int(kind) == 1:
                    ax.add_patch(Circle((x, y), hx, color="0.6"))
                else:
                    c, s = math.cos(angle), math.sin(angle)
                    corners = [(x+c*u-s*v, y+s*u+c*v) for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
                    ax.add_patch(Polygon(corners, color="0.6"))
            ax.plot(*zip(*case["reference_path"], strict=True), "--", color="0.7")
            row = selected[name]
            if row and row.get("trajectory"):
                ax.plot(*zip(*row["trajectory"], strict=True), label=name)
            ax.scatter(*case["start"][:2], marker="o", color="green")
            ax.scatter(*case["goal"], marker="*", color="red")
            ax.set(xlim=(-md.size/2, md.size/2), ylim=(-md.size/2, md.size/2), title=name, aspect="equal")
        fig.tight_layout()
        filename = f"{case['id'][:16]}.png"
        fig.savefig(report / filename)
        plt.close(fig)
        lines.extend(["", f"![{case['name']}]({filename})", ""])
    (report / "README.md").write_text("\n".join(lines) + "\n")
