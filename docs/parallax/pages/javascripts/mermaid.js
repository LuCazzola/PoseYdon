// Renders ```mermaid blocks ourselves instead of via Material's built-in
// integration: Material draws into a closed shadow DOM whose class-diagram
// markers come out dark-on-dark in the slate scheme, and can't be restyled.
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.10.0/dist/mermaid.esm.min.mjs";
import svgPanZoom from "https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.2/+esm";

// Class-diagram roles, assigned per diagram with `cssClass "A,B" <role>`.
const ROLES = {
  iface: { label: "interface / base / Protocol", light: ["#dbe9fb", "#2f6db3"], dark: ["#1d3557", "#6ea8fe"] },
  impl: { label: "implementation", light: ["#ddf3e4", "#2e8b57"], dark: ["#1e4030", "#5cc98a"] },
  data: { label: "config / data", light: ["#fdf0cf", "#b8860b"], dark: ["#4a3a12", "#e0b44c"] },
  ext: { label: "defined in another package", light: ["#efefef", "#8a8a8a"], dark: ["#33363d", "#9a9a9a"] },
};

const blocks = [...document.querySelectorAll("pre.mermaid-diagram")].map((pre) => {
  const div = document.createElement("div");
  div.className = "mermaid-diagram";
  const source = pre.textContent;
  pre.replaceWith(div);
  return { div, source };
});

function legend(source, dark) {
  const used = Object.keys(ROLES).filter((role) => new RegExp(`cssClass\\s+"[^"]*"\\s+${role}\\b`).test(source));
  if (!used.length) return "";
  const items = used.map((role) => {
    const [fill, stroke] = ROLES[role][dark ? "dark" : "light"];
    return `<span style="display:inline-flex;align-items:center;gap:.3em;margin-right:1.2em">` +
      `<span style="width:.9em;height:.9em;border:1.5px solid ${stroke};background:${fill};display:inline-block"></span>` +
      `${ROLES[role].label}</span>`;
  });
  return `<div style="font-size:.7rem;margin-top:.4em">${items.join("")}</div>`;
}

let renderId = 0;
async function render() {
  const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
  const line = dark ? "#e6e6e6" : "#444444";
  const roleCss = Object.entries(ROLES).map(([role, colors]) => {
    const [fill, stroke] = colors[dark ? "dark" : "light"];
    return `.node.${role} .label-container path { fill: ${fill} !important; stroke: ${stroke} !important; stroke-width: 1.5px !important; }`;
  }).join("\n");
  mermaid.initialize({
    startOnLoad: false,
    theme: dark ? "dark" : "default",
    themeVariables: { lineColor: line },
    // Hollow markers (inheritance, aggregation) keep the page background
    // inside, so they stay distinguishable from filled ones (composition).
    themeCSS: `.marker { stroke-width: 1.5; }
      .marker.extension, .marker.aggregation { fill: ${dark ? "#1e2129" : "#ffffff"} !important; }
      ${roleCss}`,
  });
  for (const block of blocks) {
    const { svg } = await mermaid.render(`mermaid-${renderId++}`, block.source);
    block.div.innerHTML = svg + legend(block.source, dark);
  }
}

// Click a diagram to open it full-screen: wheel zooms, drag pans, Esc or a
// click on the backdrop closes. Zoom stays out of the page so scrolling works.
function openZoom(svg) {
  const overlay = document.createElement("div");
  overlay.style.cssText = "position:fixed;inset:0;z-index:1000;display:flex;padding:2vh 2vw;" +
    "background:var(--md-default-bg-color);box-shadow:0 0 0 100vmax rgba(0,0,0,.5)";
  const copy = svg.cloneNode(true);
  copy.removeAttribute("style");  // mermaid's max-width would cap the zoom area
  copy.setAttribute("width", "100%");
  copy.setAttribute("height", "100%");
  overlay.append(copy);
  document.body.append(overlay);
  const panZoom = svgPanZoom(copy, { zoomScaleSensitivity: 0.3, maxZoom: 20, controlIconsEnabled: true });
  const close = () => { panZoom.destroy(); overlay.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
}

if (blocks.length) {
  for (const { div } of blocks) {
    div.style.cursor = "zoom-in";
    div.title = "Click to open full-screen (wheel to zoom, drag to pan, Esc to close)";
    div.addEventListener("click", () => { const svg = div.querySelector("svg"); if (svg) openZoom(svg); });
  }
  render();
  new MutationObserver(render).observe(document.body, {
    attributes: true, attributeFilter: ["data-md-color-scheme"],
  });
}
