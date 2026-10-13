// Renders ```mermaid fences (emitted as <pre class="diagram">, see mkdocs.yml)
// into the page itself, not Material's built-in integration: that one draws
// into a closed shadow DOM where class-diagram arrowheads and diamonds come out
// dark-on-dark under the slate scheme, and no page CSS can reach them.
// Mermaid's own theme follows the palette and the diagrams redraw on toggle.
//
// A diagram tags its classes with `cssClass "A,B" iface|impl|data|ext`
// (legend: docs/api/index.md); the colours are added here, per scheme, so each
// stays readable on both backgrounds, and each diagram gets a legend of the
// roles it uses, drawn from the same table.
(() => {
  const roles = {
    default: {
      iface: "fill:#dbeafe,stroke:#2563eb,color:#1e3a8a",
      impl: "fill:#dcfce7,stroke:#16a34a,color:#14532d",
      data: "fill:#fef3c7,stroke:#d97706,color:#78350f",
      ext: "fill:#f3f4f6,stroke:#9ca3af,color:#374151,stroke-dasharray:4 3",
    },
    dark: {
      iface: "fill:#1e3a5f,stroke:#60a5fa,color:#dbeafe",
      impl: "fill:#14432a,stroke:#4ade80,color:#dcfce7",
      data: "fill:#5c3a0c,stroke:#fbbf24,color:#fef3c7",
      ext: "fill:#2b313b,stroke:#9ca3af,color:#e5e7eb,stroke-dasharray:4 3",
    },
  };
  const labels = {
    iface: "interface / base class / Protocol",
    impl: "implemented here",
    data: "data class",
    ext: "defined elsewhere, for context",
  };
  // One swatch per role the diagram uses, styled like its boxes.
  function legend(source, scheme) {
    const used = new Set([...source.matchAll(/^\s*cssClass\s+"[^"]+"\s+(\w+)\s*$/gm)].map((m) => m[1]));
    const items = Object.keys(labels).filter((role) => used.has(role));
    if (!items.length) return null;
    const box = document.createElement("div");
    box.className = "diagram-legend";
    for (const role of items) {
      const css = Object.fromEntries(roles[scheme][role].split(",").map((kv) => kv.split(":")));
      const swatch = document.createElement("span");
      swatch.className = "diagram-legend__swatch";
      swatch.style.background = css.fill;
      swatch.style.border = `1px ${css["stroke-dasharray"] ? "dashed" : "solid"} ${css.stroke}`;
      const item = document.createElement("span");
      item.append(swatch, labels[role]);
      box.append(item);
    }
    return box;
  }
  // `cssClass "A,B" role` becomes one `style` line per class: class diagrams
  // in this Mermaid take `style`, but ignore `classDef` colours.
  const styled = (source, scheme) =>
    source.replace(/^(\s*)cssClass\s+"([^"]+)"\s+(\w+)\s*$/gm, (line, indent, names, role) =>
      roles[scheme][role]
        ? names.split(",").map((name) => `${indent}style ${name.trim()} ${roles[scheme][role]}`).join("\n")
        : line
    );
  const theme = () =>
    document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "default";
  let n = 0;

  async function draw() {
    const scheme = theme();
    // Natural size, not stretched to the column; Material still caps an svg at
    // the column width, so a wider one shrinks to fit and a click opens it
    // full-screen to zoom into.
    mermaid.initialize({
      startOnLoad: false,
      theme: scheme,
      securityLevel: "strict",
      class: { useMaxWidth: false },
      flowchart: { useMaxWidth: false },
    });
    for (const pre of document.querySelectorAll("pre.diagram")) {
      pre.dataset.source ??= pre.textContent;
      let out = pre.nextElementSibling;
      if (!out?.classList.contains("diagram-svg")) {
        out = document.createElement("div");
        out.className = "diagram-svg";
        pre.after(out);
      }
      const { svg } = await mermaid.render(`diagram-${n++}`, styled(pre.dataset.source, scheme));
      out.innerHTML = svg;
      if (out.nextElementSibling?.classList.contains("diagram-legend")) out.nextElementSibling.remove();
      const key = legend(pre.dataset.source, scheme);
      if (key) out.after(key);
      pre.style.display = "none";
    }
  }

  // Click a diagram to open it full-screen: the wheel zooms and dragging pans
  // there (svg-pan-zoom), never on the page itself; Esc or a click outside
  // closes it.
  function open(svg) {
    const overlay = document.createElement("div");
    overlay.className = "diagram-overlay";
    const panel = document.createElement("div");
    panel.className = "diagram-overlay__panel";
    const copy = svg.cloneNode(true);
    copy.removeAttribute("style");
    copy.setAttribute("width", "100%");
    copy.setAttribute("height", "100%");
    panel.append(copy);
    overlay.append(panel);
    document.body.append(overlay);
    const zoom = svgPanZoom(copy, { fit: true, center: true, maxZoom: 20 });
    const close = () => {
      zoom.destroy();
      overlay.remove();
      document.removeEventListener("keydown", onKey);
    };
    const onKey = (e) => e.key === "Escape" && close();
    document.addEventListener("keydown", onKey);
    overlay.addEventListener("click", (e) => e.target === overlay && close());
  }
  document.addEventListener("click", (e) => {
    const svg = e.target.closest(".diagram-svg")?.querySelector("svg");
    if (svg) open(svg);
  });

  draw();
  new MutationObserver(draw).observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
})();
