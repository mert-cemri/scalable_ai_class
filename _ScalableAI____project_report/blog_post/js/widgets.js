/* Interactive widgets for the EvoScale blog post.
   Loaded as a classic script (not a module) and relies on React, ReactDOM,
   Recharts, and htm being available as window globals. The corresponding
   <script> tags in index.html pull them from unpkg. */

(function () {
  "use strict";

  function showError(rootId, message) {
    var el = document.getElementById(rootId);
    if (!el) return;
    el.innerHTML =
      '<div style="font-family:var(--sans);font-size:13px;color:var(--miss);' +
      'padding:12px;border:1px solid var(--miss);border-radius:4px;background:#fff5f5;">' +
      message + "</div>";
  }

  // Sanity-check dependencies before doing anything.
  var missing = [];
  if (typeof window.React === "undefined") missing.push("React");
  if (typeof window.ReactDOM === "undefined") missing.push("ReactDOM");
  if (typeof window.Recharts === "undefined") missing.push("Recharts");
  if (typeof window.htm === "undefined") missing.push("htm");

  if (missing.length) {
    var msg =
      "Interactive widget couldn't load (" +
      missing.join(", ") +
      " unavailable). This usually means your browser is offline or blocking " +
      "the unpkg.com CDN. The page is still readable; the static figure below " +
      "Visual #5 shows the same data.";
    showError("kslider-root", msg);
    showError("results-root", msg);
    return;
  }

  var React = window.React;
  var ReactDOM = window.ReactDOM;
  var Recharts = window.Recharts;
  var htm = window.htm;

  var useState = React.useState;
  var useMemo = React.useMemo;
  var createElement = React.createElement;
  var createRoot = ReactDOM.createRoot;

  var LineChart = Recharts.LineChart;
  var ComposedChart = Recharts.ComposedChart;
  var Area = Recharts.Area;
  var Line = Recharts.Line;
  var XAxis = Recharts.XAxis;
  var YAxis = Recharts.YAxis;
  var CartesianGrid = Recharts.CartesianGrid;
  var Tooltip = Recharts.Tooltip;
  var Legend = Recharts.Legend;
  var ResponsiveContainer = Recharts.ResponsiveContainer;

  var html = htm.bind(createElement);

  function renderLegend(props) {
    var items = (props.payload || []).filter(function (p) { return p.type !== "none"; });
    if (!items.length) return null;
    return html`
      <ul style=${{ display: "flex", flexWrap: "wrap", gap: "12px", listStyle: "none", margin: 0, padding: "0 0 6px", fontFamily: "var(--sans)", fontSize: 12 }}>
        ${items.map(function (p) {
          return html`
            <li key=${p.dataKey} style=${{ display: "flex", alignItems: "center", gap: 5, color: "#504f4f" }}>
              <svg width="24" height="4"><line x1="0" y1="2" x2="24" y2="2" stroke=${p.color} strokeWidth="2.5" strokeDasharray=${p.payload && p.payload.strokeDasharray ? p.payload.strokeDasharray : "0"} /></svg>
              ${p.value}
            </li>`;
        })}
      </ul>`;
  }


  // ---- Visual #3: K-slider ----
  // Hard-coded from the paper. Cache hit rate is averaged across the two
  // benchmarks; intermediate K=3,5,6,7 are linearly interpolated.
  var KSWEEP = [
    { K: 1, hit: 17.8, llm: 47.1, parents: 8 },
    { K: 2, hit: 67.2, llm: 32.3, parents: 4 },
    { K: 3, hit: 72.6, llm: 23.9, parents: 3 },
    { K: 4, hit: 77.9, llm: 15.5, parents: 2 },
    { K: 5, hit: 79.8, llm: 13.9, parents: 2 },
    { K: 6, hit: 81.8, llm: 12.4, parents: 2 },
    { K: 7, hit: 83.7, llm: 10.8, parents: 2 },
    { K: 8, hit: 85.6, llm:  9.3, parents: 1 },
  ];

  var DOT_PALETTE = [
    "#003262", "#fdb515", "#1f9d55", "#c43d3d",
    "#7e57c2", "#26a69a", "#ef6c00", "#5d4037",
  ];

  function KSlider() {
    var s = useState(1);
    var K = s[0], setK = s[1];
    var row = KSWEEP.find(function (r) { return r.K === K; }) || KSWEEP[0];

    var dots = useMemo(function () {
      var out = [];
      for (var i = 0; i < 8; i++) {
        var group = Math.floor(i / K);
        out.push(DOT_PALETTE[group % DOT_PALETTE.length]);
      }
      return out;
    }, [K]);

    var hitWidth = row.hit + "%";
    var llmWidth = Math.min(100, (row.llm / 60) * 100) + "%";

    return html`
      <div>
        <div className="kslider-controls">
          <label>K =</label>
          <span className="k-value">${K}</span>
          <input
            type="range"
            min="1"
            max="8"
            step="1"
            value=${K}
            onInput=${function (e) { setK(parseInt(e.target.value, 10)); }}
            aria-label="Batch size K"
          />
        </div>

        <div className="kpanel">
          <div className="label">Cache hit rate</div>
          <div className="bar-track">
            <div className="bar-fill" style=${{ width: hitWidth, background: "var(--hit)" }}></div>
          </div>
          <div className="readout">${row.hit.toFixed(1)}%</div>
        </div>

        <div className="kpanel">
          <div className="label">LLM time / candidate</div>
          <div className="bar-track">
            <div className="bar-fill" style=${{ width: llmWidth, background: "var(--accent)" }}></div>
          </div>
          <div className="readout">${row.llm.toFixed(1)}s</div>
        </div>

        <div className="kpanel">
          <div className="label">Distinct parents (per 8 cands.)</div>
          <div className="dots-row">
            ${dots.map(function (c, i) {
      return html`<div key=${i} className="dot" style=${{ background: c }}></div>`;
    })}
          </div>
          <div className="readout">${row.parents}</div>
        </div>

        <p style=${{
        fontFamily: "var(--sans)",
        fontSize: "13px",
        color: "var(--muted)",
        marginTop: "10px",
        marginBottom: 0,
      }}>
          Drag the slider. The dots show, for every 8 candidates the system generates, how many
          distinct parents they came from. At K=1 every candidate is from a different parent.
          At K=8 they're all siblings.
        </p>
      </div>
    `;
  }

  // ---- Visual #5: best-score vs. wall-clock ----
  // Real experiment data: mean and ±1σ over 3 random seeds.
  // Scores are forward-filled with each run's final value after it ends,
  // so the mean never drops due to a high-scoring run finishing early.
  var CHART_DATA = {
    "Circle Packing": [
      { t:  15, k1: 0.9155, k1_lo: 0.8541, k1_band: 0.1228, k2: 0.7961, k2_lo: 0.6120, k2_band: 0.3683, k4: 0.9431, k4_lo: 0.8742, k4_band: 0.1258, k8: 0.9501, k8_lo: 0.8918, k8_band: 0.1082 },
      { t:  30, k1: 0.9892, k1_lo: 0.9806, k1_band: 0.0172, k2: 0.8155, k2_lo: 0.6222, k2_band: 0.3778, k4: 0.9809, k4_lo: 0.9650, k4_band: 0.0319, k8: 0.9934, k8_lo: 0.9893, k8_band: 0.0083 },
      { t:  45, k1: 0.9893, k1_lo: 0.9807, k1_band: 0.0172, k2: 0.9010, k2_lo: 0.7882, k2_band: 0.2118, k4: 0.9814, k4_lo: 0.9651, k4_band: 0.0325, k8: 0.9954, k8_lo: 0.9941, k8_band: 0.0027 },
      { t:  60, k1: 0.9893, k1_lo: 0.9807, k1_band: 0.0172, k2: 0.9831, k2_lo: 0.9791, k2_band: 0.0080, k4: 0.9833, k4_lo: 0.9655, k4_band: 0.0345, k8: 0.9960, k8_lo: 0.9954, k8_band: 0.0012 },
      { t:  75, k1: 0.9893, k1_lo: 0.9807, k1_band: 0.0172, k2: 0.9859, k2_lo: 0.9780, k2_band: 0.0159, k4: 0.9953, k4_lo: 0.9929, k4_band: 0.0047, k8: 0.9960, k8_lo: 0.9954, k8_band: 0.0012 },
      { t:  90, k1: 0.9947, k1_lo: 0.9932, k1_band: 0.0031, k2: 0.9859, k2_lo: 0.9780, k2_band: 0.0159, k4: 0.9957, k4_lo: 0.9937, k4_band: 0.0040, k8: 0.9960, k8_lo: 0.9954, k8_band: 0.0012 },
    ],
    "Signal Processing": [
      { t:  15, k1: 0.5237, k1_lo: 0.4888, k1_band: 0.0697, k2: 0.5540, k2_lo: 0.4992, k2_band: 0.1096, k4: 0.5567, k4_lo: 0.5175, k4_band: 0.0783, k8: 0.6178, k8_lo: 0.5505, k8_band: 0.1346 },
      { t:  30, k1: 0.5896, k1_lo: 0.5436, k1_band: 0.0921, k2: 0.5540, k2_lo: 0.4992, k2_band: 0.1096, k4: 0.6022, k4_lo: 0.5955, k4_band: 0.0134, k8: 0.6505, k8_lo: 0.6004, k8_band: 0.1001 },
      { t:  45, k1: 0.5902, k1_lo: 0.5444, k1_band: 0.0915, k2: 0.5638, k2_lo: 0.4972, k2_band: 0.1332, k4: 0.6198, k4_lo: 0.5947, k4_band: 0.0503, k8: 0.6607, k8_lo: 0.6054, k8_band: 0.1106 },
      { t:  60, k1: 0.5902, k1_lo: 0.5444, k1_band: 0.0915, k2: 0.5939, k2_lo: 0.5273, k2_band: 0.1333, k4: 0.6737, k4_lo: 0.6220, k4_band: 0.1035, k8: 0.6678, k8_lo: 0.6033, k8_band: 0.1289 },
      { t:  75, k1: 0.5902, k1_lo: 0.5444, k1_band: 0.0915, k2: 0.6235, k2_lo: 0.5242, k2_band: 0.1987, k4: 0.6737, k4_lo: 0.6220, k4_band: 0.1035, k8: 0.6678, k8_lo: 0.6033, k8_band: 0.1289 },
      { t:  90, k1: 0.5902, k1_lo: 0.5444, k1_band: 0.0915, k2: 0.6527, k2_lo: 0.5449, k2_band: 0.2156, k4: 0.6737, k4_lo: 0.6220, k4_band: 0.1035, k8: 0.6678, k8_lo: 0.6033, k8_band: 0.1289 },
      { t: 105, k1: 0.5902, k1_lo: 0.5444, k1_band: 0.0915, k2: 0.6540, k2_lo: 0.5462, k2_band: 0.2155, k4: 0.6737, k4_lo: 0.6220, k4_band: 0.1035, k8: 0.6678, k8_lo: 0.6033, k8_band: 0.1289 },
      { t: 120, k1: 0.5902, k1_lo: 0.5444, k1_band: 0.0915, k2: 0.6540, k2_lo: 0.5462, k2_band: 0.2155, k4: 0.6737, k4_lo: 0.6220, k4_band: 0.1035, k8: 0.6678, k8_lo: 0.6033, k8_band: 0.1289 },
    ],
  };

  var K_LINES = [
    { key: "k1", lo: "k1_lo", band: "k1_band", label: "K=1", color: "#888" },
    { key: "k2", lo: "k2_lo", band: "k2_band", label: "K=2", color: "#1f9d55" },
    { key: "k4", lo: "k4_lo", band: "k4_band", label: "K=4", color: "#003262" },
    { key: "k8", lo: "k8_lo", band: "k8_band", label: "K=8", color: "#fdb515" },
  ];

  function getYDomain(data, lines) {
    var vals = [];
    lines.forEach(function (l) {
      data.forEach(function (d) {
        var v = d[l.key];
        if (v != null && isFinite(v)) vals.push(v);
      });
    });
    if (!vals.length) return [0, 1];
    var mn = Math.min.apply(null, vals);
    var mx = Math.max.apply(null, vals);
    return [Math.max(0, mn - 0.02), Math.min(1.005, mx + 0.005)];
  }

  function ResultsChart() {
    var s = useState("Circle Packing");
    var bench = s[0], setBench = s[1];
    var data = CHART_DATA[bench];
    var yDomain = getYDomain(data, K_LINES);
    var isCP = bench === "Circle Packing";
    var xMax   = isCP ? 100 : 120;
    var xTicks = isCP ? [0, 15, 30, 45, 60, 75, 90] : [0, 15, 30, 45, 60, 75, 90, 105, 120];
    var readout = isCP
      ? "K=4 and K=8 reach their peak fastest — K=8 is at 0.99 by t=30 while K=1 arrives by t=45. All four converge by t=60. Bands are mean ± 1σ over 3 seeds."
      : "K=8 leads at t=15 but K=4 matches it by t=60 (0.674 vs 0.668). Wide bands reflect high seed-to-seed variance. Bands are mean ± 1σ over 3 seeds.";

    function customTooltip(props) {
      if (!props.active || !props.payload || !props.payload.length) return null;
      var kRows = props.payload.filter(function (p) { return p.name && p.name.match(/^K=/); });
      if (!kRows.length) return null;
      return html`
        <div style=${{ fontFamily: "var(--sans)", fontSize: 12, background: "white", border: "1px solid #ddd", padding: "8px 10px", borderRadius: 4 }}>
          <div style=${{ marginBottom: 4, fontWeight: 600 }}>t = ${props.label} min</div>
          ${kRows.map(function (p) {
            return html`<div key=${p.dataKey} style=${{ color: p.color, margin: "2px 0" }}>
              ${p.name}: ${typeof p.value === "number" ? p.value.toFixed(4) : "—"}
            </div>`;
          })}
        </div>`;
    }

    return html`
      <div>
        <div className="toggle-row">
          ${Object.keys(CHART_DATA).map(function (name) {
            return html`
              <button
                key=${name}
                className=${name === bench ? "active" : ""}
                onClick=${function () { setBench(name); }}
              >${name}</button>
            `;
          })}
        </div>

        <div className="chart-wrap">
          <${ResponsiveContainer} width="100%" height="100%">
            <${ComposedChart} data=${data} margin=${{ top: 10, right: 16, left: 0, bottom: 28 }}>
              <${CartesianGrid} strokeDasharray="3 3" stroke="#eee" />
              <${XAxis}
                dataKey="t"
                type="number"
                domain=${[0, xMax]}
                ticks=${xTicks}
                tick=${{ fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" }}
                label=${{
                  value: "Wall-clock time (minutes)",
                  position: "insideBottom",
                  offset: -16,
                  style: { fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" },
                }}
              />
              <${YAxis}
                type="number"
                domain=${yDomain}
                allowDataOverflow=${true}
                tick=${{ fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" }}
                label=${{
                  value: "Best score",
                  angle: -90,
                  position: "insideLeft",
                  offset: 16,
                  style: { fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" },
                }}
              />
              <${Tooltip} content=${customTooltip} />
              <${Legend}
                verticalAlign="top"
                height=${28}
                content=${renderLegend}
              />
              ${K_LINES.map(function (l) {
                return html`
                  <${Area}
                    key=${l.key + "_lo"}
                    type="monotone"
                    dataKey=${l.lo}
                    stackId=${l.key}
                    stroke="none"
                    fill="transparent"
                    legendType="none"
                    connectNulls=${false}
                    isAnimationActive=${false}
                    dot=${false}
                  />
                  <${Area}
                    key=${l.key + "_band"}
                    type="monotone"
                    dataKey=${l.band}
                    stackId=${l.key}
                    stroke="none"
                    fill=${l.color}
                    fillOpacity=${0.18}
                    legendType="none"
                    connectNulls=${false}
                    isAnimationActive=${false}
                    dot=${false}
                  />
                  <${Line}
                    key=${l.key}
                    type="monotone"
                    dataKey=${l.key}
                    stroke=${l.color}
                    strokeWidth=${2.2}
                    strokeDasharray=${l.dashed ? "6 3" : "0"}
                    dot=${{ r: 3 }}
                    name=${l.label}
                    connectNulls=${false}
                    isAnimationActive=${true}
                  />
                `;
              })}
            <//>
          <//>
        </div>

        <div className="chart-readout">${readout}</div>
      </div>
    `;
  }

  // ---- Speculative pipelining — candidates chart ----
  // All four conditions on one chart; mean ± 1σ over 3 runs.
  var SPEC_CAND_DATA = [
    { t:  0, k4:   0.0, k4_lo:   0.0, k4_band:  0.0, k4s:   0.0, k4s_lo:   0.0, k4s_band:  0.0, k8:   0.0, k8_lo:   0.0, k8_band:  0.0, k8s:   0.0, k8s_lo:   0.0, k8s_band:  0.0 },
    { t: 15, k4:  24.0, k4_lo:  18.3, k4_band: 11.4, k4s:  28.0, k4s_lo:  24.7, k4s_band:  6.6, k8:  40.0, k8_lo:  33.5, k8_band: 13.0, k8s:  42.7, k8s_lo:  38.9, k8s_band:  7.5 },
    { t: 30, k4:  34.7, k4_lo:  27.9, k4_band: 13.6, k4s:  41.3, k4s_lo:  39.4, k4s_band:  3.8, k8:  53.3, k8_lo:  49.6, k8_band:  7.5, k8s:  66.7, k8s_lo:  56.7, k8s_band: 19.9 },
    { t: 45, k4:  44.0, k4_lo:  37.5, k4_band: 13.0, k4s:  58.7, k4s_lo:  51.9, k4s_band: 13.6, k8:  72.0, k8_lo:  65.5, k8_band: 13.0, k8s:  80.0, k8s_lo:  66.9, k8s_band: 26.2 },
    { t: 60, k4:  56.0, k4_lo:  44.2, k4_band: 23.6, k4s:  70.7, k4s_lo:  59.2, k4s_band: 22.9, k8:  85.3, k8_lo:  81.6, k8_band:  7.5, k8s: 104.0, k8s_lo:  90.9, k8s_band: 26.2 },
    { t: 75, k4:  66.7, k4_lo:  57.2, k4_band: 18.9, k4s:  96.0, k4s_lo:  77.8, k4s_band: 36.4, k8: 101.3, k8_lo:  97.6, k8_band:  7.5, k8s: 125.3, k8s_lo: 104.3, k8s_band: 42.0 },
    { t: 90, k4:  84.0, k4_lo:  77.5, k4_band: 13.0, k4s: 110.7, k4s_lo:  92.7, k4s_band: 36.0, k8: 112.0, k8_lo: 112.0, k8_band:  0.0, k8s: 136.0, k8s_lo: 118.7, k8s_band: 34.6 },
  ];

  var SPEC_CAND_LINES = [
    { key: "k4",  lo: "k4_lo",  band: "k4_band",  label: "K=4",        color: "#2ca02c", dashed: false },
    { key: "k4s", lo: "k4s_lo", band: "k4s_band", label: "K=4 (spec)", color: "#2ca02c", dashed: true  },
    { key: "k8",  lo: "k8_lo",  band: "k8_band",  label: "K=8",        color: "#d62728", dashed: false },
    { key: "k8s", lo: "k8s_lo", band: "k8s_band", label: "K=8 (spec)", color: "#d62728", dashed: true  },
  ];

  function SpecCandChart() {
    function customTooltip(props) {
      if (!props.active || !props.payload || !props.payload.length) return null;
      var validNames = { "K=4": 1, "K=4 (spec)": 1, "K=8": 1, "K=8 (spec)": 1 };
      var rows = props.payload.filter(function (p) { return p.name && validNames[p.name]; });
      if (!rows.length) return null;
      return html`
        <div style=${{ fontFamily: "var(--sans)", fontSize: 12, background: "white", border: "1px solid #ddd", padding: "8px 10px", borderRadius: 4 }}>
          <div style=${{ marginBottom: 4, fontWeight: 600 }}>t = ${props.label} min</div>
          ${rows.map(function (p) {
            return html`<div key=${p.dataKey} style=${{ color: p.color, margin: "2px 0" }}>
              ${p.name}: ${typeof p.value === "number" ? Math.round(p.value) : "—"}
            </div>`;
          })}
        </div>`;
    }

    return html`
      <div>
        <div className="chart-wrap">
          <${ResponsiveContainer} width="100%" height="100%">
            <${ComposedChart} data=${SPEC_CAND_DATA} margin=${{ top: 10, right: 16, left: 0, bottom: 28 }}>
              <${CartesianGrid} strokeDasharray="3 3" stroke="#eee" />
              <${XAxis}
                dataKey="t"
                type="number"
                domain=${[0, 90]}
                ticks=${[0, 15, 30, 45, 60, 75, 90]}
                tick=${{ fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" }}
                label=${{
                  value: "Wall-clock time (minutes)",
                  position: "insideBottom",
                  offset: -16,
                  style: { fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" },
                }}
              />
              <${YAxis}
                type="number"
                domain=${[0, "auto"]}
                tick=${{ fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" }}
                label=${{
                  value: "Candidates generated",
                  angle: -90,
                  position: "insideLeft",
                  offset: 16,
                  style: { fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" },
                }}
              />
              <${Tooltip} content=${customTooltip} />
              <${Legend}
                verticalAlign="top"
                height=${28}
                content=${renderLegend}
              />
              ${SPEC_CAND_LINES.map(function (l) {
                return html`
                  <${Area}
                    key=${l.key + "_lo"}
                    type="monotone"
                    dataKey=${l.lo}
                    stackId=${l.key}
                    stroke="none"
                    fill="transparent"
                    legendType="none"
                    isAnimationActive=${false}
                    dot=${false}
                  />
                  <${Area}
                    key=${l.key + "_band"}
                    type="monotone"
                    dataKey=${l.band}
                    stackId=${l.key}
                    stroke="none"
                    fill=${l.color}
                    fillOpacity=${0.18}
                    legendType="none"
                    isAnimationActive=${false}
                    dot=${false}
                  />
                  <${Line}
                    key=${l.key}
                    type="monotone"
                    dataKey=${l.key}
                    stroke=${l.color}
                    strokeWidth=${2.2}
                    strokeDasharray=${l.dashed ? "6 3" : "0"}
                    dot=${{ r: 3 }}
                    name=${l.label}
                    isAnimationActive=${true}
                  />
                `;
              })}
            <//>
          <//>
        </div>
        <div className="chart-readout">
          Spec variants generate more candidates per hour across both K values.
          K=8 (spec) reaches ~136 candidates in 90 min vs. ~112 without speculation.
          Bands are mean ± 1σ over 3 seeds.
        </div>
      </div>
    `;
  }

  // ---- Speculation hit timeline data ----
  // Per-iteration records from circle_packing spec runs (deduplicated by iteration).
  // Fields: i=iteration index, hit=true/false, llm=LLM gen time (s),
  //         iter=total iter time (s), best=global best score, kv=KV cache hit rate.
  var SPEC_HIT_DATA = {"K=4": {"run1": [{"i": 1, "hit": false, "llm": 100.86, "iter": 101.13, "best": 0.36424, "kv": 0.7309}, {"i": 2, "hit": false, "llm": 42.03, "iter": 42.34, "best": 0.50419, "kv": 0.703}, {"i": 3, "hit": false, "llm": 40.69, "iter": 40.99, "best": 0.50419, "kv": 0.839}, {"i": 4, "hit": false, "llm": 124.57, "iter": 168.46, "best": 0.99166, "kv": 0.7475}, {"i": 5, "hit": false, "llm": 91.57, "iter": 181.29, "best": 0.99166, "kv": 0.7775}, {"i": 6, "hit": false, "llm": 61.22, "iter": 421.24, "best": 0.99166, "kv": 0.8038}, {"i": 7, "hit": true, "llm": 0.01, "iter": 24.83, "best": 0.99166, "kv": 0}, {"i": 8, "hit": false, "llm": 112.16, "iter": 344.89, "best": 0.99604, "kv": 0.7634}, {"i": 9, "hit": true, "llm": 0.01, "iter": 360.13, "best": 0.99604, "kv": 0}, {"i": 10, "hit": false, "llm": 112.8, "iter": 135.89, "best": 0.99604, "kv": 0.7779}, {"i": 11, "hit": false, "llm": 99.58, "iter": 459.63, "best": 0.99604, "kv": 0.7855}, {"i": 12, "hit": true, "llm": 0.01, "iter": 360.04, "best": 0.99604, "kv": 0}, {"i": 13, "hit": false, "llm": 132.55, "iter": 133.4, "best": 0.99604, "kv": 0.742}, {"i": 14, "hit": true, "llm": 95.12, "iter": 618.28, "best": 0.99604, "kv": 0}, {"i": 15, "hit": false, "llm": 125.65, "iter": 485.78, "best": 0.99604, "kv": 0.7452}, {"i": 16, "hit": false, "llm": 91.46, "iter": 118.65, "best": 0.99604, "kv": 0.7935}, {"i": 17, "hit": true, "llm": 113.65, "iter": 130.27, "best": 0.99604, "kv": 0}, {"i": 18, "hit": false, "llm": 86.67, "iter": 160.31, "best": 0.99604, "kv": 0.7731}, {"i": 19, "hit": true, "llm": 18.9, "iter": 19.3, "best": 0.99604, "kv": 0}, {"i": 20, "hit": false, "llm": 118.35, "iter": 478.48, "best": 0.99604, "kv": 0.763}, {"i": 21, "hit": true, "llm": 0.02, "iter": 71.86, "best": 0.99604, "kv": 0}, {"i": 22, "hit": false, "llm": 103.89, "iter": 438.69, "best": 0.99604, "kv": 0.7539}, {"i": 23, "hit": true, "llm": 0.01, "iter": 34.42, "best": 0.99604, "kv": 0}, {"i": 24, "hit": false, "llm": 15.54, "iter": 15.57, "best": 0.99604, "kv": 0}, {"i": 25, "hit": true, "llm": 101.14, "iter": 402.6, "best": 0.99604, "kv": 0.9162}, {"i": 26, "hit": false, "llm": 115.85, "iter": 182.15, "best": 0.99604, "kv": 0.9114}, {"i": 27, "hit": true, "llm": 63.12, "iter": 687.7, "best": 0.99604, "kv": 0}, {"i": 28, "hit": false, "llm": 71.26, "iter": 112.18, "best": 0.99604, "kv": 0.7501}, {"i": 29, "hit": true, "llm": 80.9, "iter": 397.22, "best": 0.99604, "kv": 0}, {"i": 30, "hit": false, "llm": 89.07, "iter": 449.14, "best": 0.99604, "kv": 0.7833}], "run2": [{"i": 1, "hit": false, "llm": 59.8, "iter": 60.2, "best": 0.75943, "kv": 0.7784}, {"i": 2, "hit": false, "llm": 67.26, "iter": 67.63, "best": 0.75943, "kv": 0.7084}, {"i": 3, "hit": false, "llm": 76.34, "iter": 76.79, "best": 0.8282, "kv": 0.8382}, {"i": 4, "hit": true, "llm": 40.75, "iter": 67.23, "best": 0.8282, "kv": 0}, {"i": 5, "hit": false, "llm": 220.11, "iter": 220.83, "best": 0.82879, "kv": 0.7675}, {"i": 6, "hit": false, "llm": 112.99, "iter": 192.9, "best": 0.82879, "kv": 0.8122}, {"i": 7, "hit": true, "llm": 17.47, "iter": 109.22, "best": 0.9804, "kv": 0}, {"i": 8, "hit": false, "llm": 75.4, "iter": 78.19, "best": 0.9804, "kv": 0.7439}, {"i": 9, "hit": false, "llm": 119.73, "iter": 387.92, "best": 0.98894, "kv": 0.7497}, {"i": 10, "hit": false, "llm": 98.99, "iter": 104.65, "best": 0.99261, "kv": 0.7666}, {"i": 11, "hit": false, "llm": 81.77, "iter": 83.35, "best": 0.99261, "kv": 0.7746}, {"i": 12, "hit": false, "llm": 130.97, "iter": 527.69, "best": 0.99261, "kv": 0.7687}, {"i": 13, "hit": true, "llm": 0.02, "iter": 10.81, "best": 0.99261, "kv": 0}, {"i": 14, "hit": false, "llm": 100.21, "iter": 118.35, "best": 0.99261, "kv": 0.7579}, {"i": 15, "hit": true, "llm": 105.38, "iter": 142.02, "best": 0.99261, "kv": 0}, {"i": 16, "hit": false, "llm": 139.66, "iter": 154.48, "best": 0.99261, "kv": 0.7413}, {"i": 17, "hit": false, "llm": 140.77, "iter": 143.07, "best": 0.99511, "kv": 0.7454}, {"i": 18, "hit": false, "llm": 113.71, "iter": 500.69, "best": 0.99511, "kv": 0.7491}, {"i": 19, "hit": true, "llm": 0.01, "iter": 2.16, "best": 0.99511, "kv": 0}, {"i": 20, "hit": false, "llm": 15.54, "iter": 15.58, "best": 0.99605, "kv": 0}, {"i": 21, "hit": false, "llm": 125.61, "iter": 485.76, "best": 0.99605, "kv": 0.7385}, {"i": 22, "hit": false, "llm": 118.09, "iter": 126.36, "best": 0.99605, "kv": 0.9568}, {"i": 23, "hit": false, "llm": 116.28, "iter": 116.73, "best": 0.99605, "kv": 0.9542}, {"i": 24, "hit": false, "llm": 119.38, "iter": 144.28, "best": 0.9968, "kv": 0.7348}, {"i": 25, "hit": false, "llm": 15.55, "iter": 15.6, "best": 0.9968, "kv": 0}, {"i": 26, "hit": false, "llm": 138.0, "iter": 173.22, "best": 0.9968, "kv": 0.7359}, {"i": 27, "hit": true, "llm": 60.29, "iter": 122.97, "best": 0.9968, "kv": 0}, {"i": 28, "hit": false, "llm": 15.51, "iter": 15.54, "best": 0.9968, "kv": 0}, {"i": 29, "hit": false, "llm": 15.59, "iter": 15.63, "best": 0.9968, "kv": 0}, {"i": 30, "hit": false, "llm": 87.15, "iter": 134.85, "best": 0.9968, "kv": 0.7604}, {"i": 31, "hit": false, "llm": 91.42, "iter": 243.75, "best": 0.9968, "kv": 0.7327}, {"i": 32, "hit": false, "llm": 83.03, "iter": 145.89, "best": 0.9968, "kv": 0.7562}, {"i": 33, "hit": false, "llm": 109.49, "iter": 501.61, "best": 0.9968, "kv": 0.753}, {"i": 34, "hit": false, "llm": 75.69, "iter": 113.39, "best": 0.9968, "kv": 0.7343}, {"i": 35, "hit": false, "llm": 70.16, "iter": 240.49, "best": 0.9968, "kv": 0.6458}, {"i": 36, "hit": false, "llm": 115.28, "iter": 322.61, "best": 0.9975, "kv": 0.7326}, {"i": 37, "hit": false, "llm": 101.95, "iter": 410.25, "best": 0.9975, "kv": 0.8625}, {"i": 38, "hit": false, "llm": 90.19, "iter": 425.69, "best": 0.9975, "kv": 0.7206}, {"i": 39, "hit": false, "llm": 100.98, "iter": 197.25, "best": 0.9975, "kv": 0.7588}], "run3": [{"i": 1, "hit": false, "llm": 110.81, "iter": 111.09, "best": 0.82732, "kv": 0.7298}, {"i": 2, "hit": false, "llm": 43.65, "iter": 43.96, "best": 0.9256, "kv": 0.7398}, {"i": 3, "hit": false, "llm": 34.96, "iter": 35.28, "best": 0.9256, "kv": 0.8109}, {"i": 4, "hit": false, "llm": 93.78, "iter": 220.99, "best": 0.9256, "kv": 0.7009}, {"i": 5, "hit": true, "llm": 0.01, "iter": 360.12, "best": 0.9256, "kv": 0}, {"i": 6, "hit": false, "llm": 105.11, "iter": 105.4, "best": 0.9256, "kv": 0.7939}, {"i": 7, "hit": true, "llm": 92.92, "iter": 222.18, "best": 0.9256, "kv": 0}, {"i": 8, "hit": false, "llm": 61.55, "iter": 421.67, "best": 0.99329, "kv": 0.6867}, {"i": 9, "hit": true, "llm": 0.01, "iter": 11.37, "best": 0.99329, "kv": 0}, {"i": 10, "hit": false, "llm": 57.67, "iter": 87.2, "best": 0.99329, "kv": 0.7641}, {"i": 11, "hit": true, "llm": 80.16, "iter": 440.28, "best": 0.99329, "kv": 0}, {"i": 12, "hit": false, "llm": 73.13, "iter": 107.92, "best": 0.99329, "kv": 0.7978}, {"i": 13, "hit": false, "llm": 78.4, "iter": 79.29, "best": 0.99329, "kv": 0.7538}, {"i": 14, "hit": true, "llm": 77.57, "iter": 545.94, "best": 0.99329, "kv": 0}, {"i": 15, "hit": false, "llm": 67.27, "iter": 70.86, "best": 0.99329, "kv": 0.7367}, {"i": 16, "hit": false, "llm": 60.42, "iter": 291.14, "best": 0.99329, "kv": 0.7497}, {"i": 17, "hit": false, "llm": 90.58, "iter": 121.85, "best": 0.99329, "kv": 0.785}, {"i": 18, "hit": false, "llm": 88.75, "iter": 108.04, "best": 0.99329, "kv": 0.7586}, {"i": 19, "hit": false, "llm": 86.67, "iter": 426.53, "best": 0.99329, "kv": 0.7762}, {"i": 20, "hit": false, "llm": 152.02, "iter": 184.43, "best": 0.99329, "kv": 0.7598}, {"i": 21, "hit": true, "llm": 49.04, "iter": 67.54, "best": 0.99329, "kv": 0}, {"i": 22, "hit": false, "llm": 83.98, "iter": 102.53, "best": 0.99329, "kv": 0.8949}, {"i": 23, "hit": false, "llm": 201.75, "iter": 243.93, "best": 0.99329, "kv": 0.7522}, {"i": 24, "hit": true, "llm": 170.5, "iter": 530.61, "best": 0.99329, "kv": 0}, {"i": 25, "hit": false, "llm": 118.71, "iter": 478.91, "best": 0.99329, "kv": 0.8792}, {"i": 26, "hit": false, "llm": 226.53, "iter": 227.24, "best": 0.99329, "kv": 0.895}, {"i": 27, "hit": false, "llm": 152.99, "iter": 513.12, "best": 0.99329, "kv": 0.7315}, {"i": 28, "hit": true, "llm": 0.01, "iter": 360.05, "best": 0.99329, "kv": 0}, {"i": 29, "hit": false, "llm": 94.18, "iter": 139.91, "best": 0.99789, "kv": 0.7704}, {"i": 30, "hit": false, "llm": 113.09, "iter": 113.86, "best": 0.99789, "kv": 0.7774}, {"i": 31, "hit": true, "llm": 85.05, "iter": 115.49, "best": 0.99789, "kv": 0}, {"i": 32, "hit": false, "llm": 116.1, "iter": 124.03, "best": 0.99789, "kv": 0.7621}, {"i": 33, "hit": false, "llm": 94.53, "iter": 96.95, "best": 0.99789, "kv": 0.779}, {"i": 34, "hit": true, "llm": 71.5, "iter": 88.48, "best": 0.99789, "kv": 0}]}, "K=8": {"run1": [{"i": 1, "hit": false, "llm": 69.62, "iter": 70.19, "best": 0.51907, "kv": 0.8662}, {"i": 2, "hit": false, "llm": 64.32, "iter": 64.82, "best": 0.6327, "kv": 0.7869}, {"i": 3, "hit": true, "llm": 64.02, "iter": 64.82, "best": 0.66883, "kv": 0}, {"i": 4, "hit": false, "llm": 222.49, "iter": 437.14, "best": 0.66883, "kv": 0.8332}, {"i": 5, "hit": true, "llm": 0.01, "iter": 0.97, "best": 0.66883, "kv": 0}, {"i": 6, "hit": false, "llm": 104.93, "iter": 353.51, "best": 0.80849, "kv": 0.8314}, {"i": 7, "hit": false, "llm": 161.13, "iter": 278.57, "best": 0.80849, "kv": 0.8754}, {"i": 8, "hit": false, "llm": 129.89, "iter": 134.29, "best": 0.80849, "kv": 0.8336}, {"i": 9, "hit": false, "llm": 130.1, "iter": 490.15, "best": 0.821, "kv": 0.8439}, {"i": 10, "hit": false, "llm": 76.28, "iter": 210.15, "best": 0.92172, "kv": 0.852}, {"i": 11, "hit": false, "llm": 120.67, "iter": 692.02, "best": 0.92172, "kv": 0.8745}, {"i": 12, "hit": false, "llm": 124.72, "iter": 152.98, "best": 0.95839, "kv": 0.8823}, {"i": 13, "hit": true, "llm": 86.37, "iter": 446.49, "best": 0.95839, "kv": 0}, {"i": 14, "hit": false, "llm": 100.38, "iter": 460.57, "best": 0.95839, "kv": 0.8228}, {"i": 15, "hit": false, "llm": 115.01, "iter": 124.36, "best": 0.95839, "kv": 0.837}, {"i": 16, "hit": true, "llm": 115.45, "iter": 138.71, "best": 0.95839, "kv": 0}, {"i": 17, "hit": false, "llm": 109.95, "iter": 207.9, "best": 0.95839, "kv": 0.84}, {"i": 18, "hit": false, "llm": 96.18, "iter": 756.69, "best": 0.95839, "kv": 0.8598}, {"i": 19, "hit": false, "llm": 92.85, "iter": 518.02, "best": 0.95841, "kv": 0.8479}, {"i": 20, "hit": false, "llm": 104.13, "iter": 492.58, "best": 0.95841, "kv": 0.8446}, {"i": 21, "hit": false, "llm": 128.61, "iter": 456.14, "best": 0.95841, "kv": 0.8508}, {"i": 22, "hit": true, "llm": 0.01, "iter": 0.5, "best": 0.95841, "kv": 0}, {"i": 23, "hit": false, "llm": 15.93, "iter": 15.97, "best": 0.95841, "kv": 0}, {"i": 24, "hit": true, "llm": 117.35, "iter": 405.52, "best": 0.95841, "kv": 0.8492}, {"i": 25, "hit": false, "llm": 16.28, "iter": 16.32, "best": 0.95841, "kv": 0}], "run2": [{"i": 1, "hit": false, "llm": 80.12, "iter": 80.54, "best": 0.41152, "kv": 0.8662}, {"i": 2, "hit": false, "llm": 51.97, "iter": 412.08, "best": 0.79696, "kv": 0.7365}, {"i": 3, "hit": false, "llm": 59.37, "iter": 59.98, "best": 0.83497, "kv": 0.8142}, {"i": 4, "hit": true, "llm": 31.42, "iter": 60.79, "best": 0.83497, "kv": 0}, {"i": 5, "hit": false, "llm": 118.62, "iter": 119.57, "best": 0.83497, "kv": 0.7929}, {"i": 6, "hit": true, "llm": 69.6, "iter": 429.71, "best": 0.83497, "kv": 0}, {"i": 7, "hit": false, "llm": 104.31, "iter": 609.91, "best": 0.83497, "kv": 0.8738}, {"i": 8, "hit": false, "llm": 85.26, "iter": 588.89, "best": 0.95516, "kv": 0.8798}, {"i": 9, "hit": false, "llm": 59.73, "iter": 419.84, "best": 0.95516, "kv": 0.8214}, {"i": 10, "hit": false, "llm": 59.91, "iter": 450.75, "best": 0.95516, "kv": 0.8212}, {"i": 11, "hit": true, "llm": 0.02, "iter": 360.04, "best": 0.95516, "kv": 0}, {"i": 12, "hit": false, "llm": 97.5, "iter": 739.45, "best": 0.95516, "kv": 0.8179}, {"i": 13, "hit": false, "llm": 113.28, "iter": 473.31, "best": 0.95611, "kv": 0.868}, {"i": 14, "hit": true, "llm": 0.01, "iter": 180.25, "best": 0.95611, "kv": 0}, {"i": 15, "hit": false, "llm": 126.75, "iter": 486.88, "best": 0.95611, "kv": 0.8612}, {"i": 16, "hit": false, "llm": 114.22, "iter": 236.99, "best": 0.95836, "kv": 0.8686}, {"i": 17, "hit": false, "llm": 75.96, "iter": 108.5, "best": 0.9584, "kv": 0.8396}, {"i": 18, "hit": false, "llm": 119.43, "iter": 239.41, "best": 0.9584, "kv": 0.8848}, {"i": 19, "hit": false, "llm": 154.27, "iter": 558.1, "best": 0.9584, "kv": 0.8389}], "run3": [{"i": 1, "hit": false, "llm": 58.72, "iter": 59.41, "best": 0.36424, "kv": 0.8662}, {"i": 2, "hit": false, "llm": 48.7, "iter": 49.29, "best": 0.73897, "kv": 0.7476}, {"i": 3, "hit": false, "llm": 140.58, "iter": 268.36, "best": 0.78937, "kv": 0.884}, {"i": 4, "hit": true, "llm": 35.78, "iter": 36.49, "best": 0.78937, "kv": 0}, {"i": 5, "hit": false, "llm": 108.13, "iter": 108.64, "best": 0.79696, "kv": 0.8042}, {"i": 6, "hit": true, "llm": 103.41, "iter": 149.43, "best": 0.79696, "kv": 0}, {"i": 7, "hit": false, "llm": 124.52, "iter": 554.38, "best": 0.98611, "kv": 0.8751}, {"i": 8, "hit": false, "llm": 106.45, "iter": 109.87, "best": 0.98611, "kv": 0.827}, {"i": 9, "hit": true, "llm": 38.99, "iter": 79.56, "best": 0.98611, "kv": 0}, {"i": 10, "hit": false, "llm": 64.76, "iter": 424.87, "best": 0.99121, "kv": 0.8759}, {"i": 11, "hit": false, "llm": 113.21, "iter": 437.06, "best": 0.99121, "kv": 0.8817}, {"i": 12, "hit": false, "llm": 91.07, "iter": 451.2, "best": 0.99121, "kv": 0.7955}, {"i": 13, "hit": false, "llm": 75.19, "iter": 472.95, "best": 0.99121, "kv": 0.8391}, {"i": 14, "hit": true, "llm": 0.01, "iter": 186.27, "best": 0.99121, "kv": 0}, {"i": 15, "hit": false, "llm": 71.71, "iter": 100.82, "best": 0.99121, "kv": 0.8127}, {"i": 16, "hit": false, "llm": 72.35, "iter": 432.38, "best": 0.99121, "kv": 0.8009}, {"i": 17, "hit": false, "llm": 125.26, "iter": 485.37, "best": 0.99121, "kv": 0.8722}, {"i": 18, "hit": false, "llm": 79.88, "iter": 80.38, "best": 0.99121, "kv": 0.8779}, {"i": 19, "hit": false, "llm": 76.76, "iter": 436.81, "best": 0.99121, "kv": 0.8863}, {"i": 20, "hit": false, "llm": 228.51, "iter": 948.06, "best": 0.99121, "kv": 0.8296}, {"i": 21, "hit": true, "llm": 0.01, "iter": 39.65, "best": 0.99121, "kv": 0}, {"i": 22, "hit": false, "llm": 97.2, "iter": 457.34, "best": 0.99121, "kv": 0.8258}, {"i": 23, "hit": false, "llm": 93.7, "iter": 453.82, "best": 0.99121, "kv": 0.8564}, {"i": 24, "hit": false, "llm": 68.57, "iter": 224.9, "best": 0.99121, "kv": 0.8386}]}};

  // ---- Speculative pipelining chart ----
  // circle_packing only; mean ± 1σ over 3 runs; forward-filled.
  var SPEC_CHART_DATA = {
    "K=4": [
      { t:  0, base: 0.3642, base_lo: 0.3642, base_band: 0.0,    spec: 0.3642, spec_lo: 0.3642, spec_band: 0.0    },
      { t: 15, base: 0.9431, base_lo: 0.8742, base_band: 0.1258, spec: 0.9702, spec_lo: 0.9385, spec_band: 0.0615 },
      { t: 30, base: 0.9809, base_lo: 0.9650, base_band: 0.0319, spec: 0.9940, spec_lo: 0.9925, spec_band: 0.0030 },
      { t: 45, base: 0.9814, base_lo: 0.9651, base_band: 0.0325, spec: 0.9948, spec_lo: 0.9937, spec_band: 0.0023 },
      { t: 60, base: 0.9833, base_lo: 0.9655, base_band: 0.0345, spec: 0.9951, spec_lo: 0.9938, spec_band: 0.0026 },
      { t: 75, base: 0.9953, base_lo: 0.9929, base_band: 0.0047, spec: 0.9954, spec_lo: 0.9939, spec_band: 0.0030 },
      { t: 90, base: 0.9957, base_lo: 0.9937, base_band: 0.0040, spec: 0.9954, spec_lo: 0.9939, spec_band: 0.0030 },
    ],
    "K=8": [
      { t:  0, base: 0.3642, base_lo: 0.3642, base_band: 0.0,    spec: 0.3642, spec_lo: 0.3642, spec_band: 0.0,    fix: 0.3642, fix_lo: 0.3642, fix_band: 0.0    },
      { t: 15, base: 0.9501, base_lo: 0.8918, base_band: 0.1082, spec: 0.8765, spec_lo: 0.7983, spec_band: 0.1565, fix: 0.9153, fix_lo: 0.865,  fix_band: 0.1006 },
      { t: 30, base: 0.9934, base_lo: 0.9893, base_band: 0.0083, spec: 0.9225, spec_lo: 0.8492, spec_band: 0.1465, fix: 0.9663, fix_lo: 0.929,  fix_band: 0.0747 },
      { t: 45, base: 0.9954, base_lo: 0.9941, base_band: 0.0027, spec: 0.9560, spec_lo: 0.9276, spec_band: 0.0568, fix: 0.9692, fix_lo: 0.9298, fix_band: 0.0788 },
      { t: 60, base: 0.9960, base_lo: 0.9954, base_band: 0.0012, spec: 0.9683, spec_lo: 0.9520, spec_band: 0.0325, fix: 0.9842, fix_lo: 0.9645, fix_band: 0.0394 },
      { t: 75, base: 0.9960, base_lo: 0.9954, base_band: 0.0012, spec: 0.9686, spec_lo: 0.9525, spec_band: 0.0321, fix: 0.9846, fix_lo: 0.9654, fix_band: 0.0383 },
      { t: 90, base: 0.9960, base_lo: 0.9954, base_band: 0.0012, spec: 0.9686, spec_lo: 0.9525, spec_band: 0.0321, fix: 0.9849, fix_lo: 0.9662, fix_band: 0.0375 },
      { t:105, base: 0.9961, base_lo: 0.9955, base_band: 0.0012, spec: 0.9686, spec_lo: 0.9525, spec_band: 0.0321, fix: 0.9985, fix_lo: 0.9979, fix_band: 0.0012 },
      { t:120, base: 0.9961, base_lo: 0.9955, base_band: 0.0012, spec: 0.9686, spec_lo: 0.9525, spec_band: 0.0321, fix: 0.9989, fix_lo: 0.9978, fix_band: 0.0022 },
    ],
  };

  var SPEC_LINES = {
    "K=4": [
      { key: "base", lo: "base_lo", band: "base_band", label: "K=4",                   color: "#2ca02c", dashed: false },
      { key: "spec", lo: "spec_lo", band: "spec_band", label: "K=4 (spec)",             color: "#9467bd", dashed: false },
    ],
    "K=8": [
      { key: "base", lo: "base_lo", band: "base_band", label: "K=8",                   color: "#d62728", dashed: false },
      { key: "spec", lo: "spec_lo", band: "spec_band", label: "K=8 (spec)",             color: "#9467bd", dashed: false },
      { key: "fix",  lo: "fix_lo",  band: "fix_band",  label: "K=8 (added children)",  color: "#ff7f0e", dashed: false },
    ],
  };

  var SPEC_XMAX = { "K=4": 90, "K=8": 120 };

  function SpecChart() {
    var s = useState("K=4");
    var ksel = s[0], setKsel = s[1];
    var data  = SPEC_CHART_DATA[ksel];
    var lines = SPEC_LINES[ksel];
    var yDomain = getYDomain(data, lines);
    var xMax   = SPEC_XMAX[ksel] || 90;
    var xTicks = [];
    for (var _t = 0; _t <= xMax; _t += 15) xTicks.push(_t);

    function customTooltip(props) {
      if (!props.active || !props.payload || !props.payload.length) return null;
      var rows = props.payload.filter(function (p) { return p.name && (p.name === "K=4" || p.name === "K=8" || p.name === "K=4 (spec)" || p.name === "K=8 (spec)" || p.name === "K=8 (added children)"); });
      if (!rows.length) return null;
      return html`
        <div style=${{ fontFamily: "var(--sans)", fontSize: 12, background: "white", border: "1px solid #ddd", padding: "8px 10px", borderRadius: 4 }}>
          <div style=${{ marginBottom: 4, fontWeight: 600 }}>t = ${props.label} min</div>
          ${rows.map(function (p) {
            return html`<div key=${p.dataKey} style=${{ color: p.color, margin: "2px 0" }}>
              ${p.name}: ${typeof p.value === "number" ? p.value.toFixed(4) : "—"}
            </div>`;
          })}
        </div>`;
    }

    return html`
      <div>
        <div className="toggle-row">
          ${Object.keys(SPEC_CHART_DATA).map(function (name) {
            return html`
              <button
                key=${name}
                className=${name === ksel ? "active" : ""}
                onClick=${function () { setKsel(name); }}
              >${name}</button>
            `;
          })}
        </div>

        <div className="chart-wrap">
          <${ResponsiveContainer} width="100%" height="100%">
            <${ComposedChart} data=${data} margin=${{ top: 10, right: 16, left: 0, bottom: 28 }}>
              <${CartesianGrid} strokeDasharray="3 3" stroke="#eee" />
              <${XAxis}
                dataKey="t"
                type="number"
                domain=${[0, xMax]}
                ticks=${xTicks}
                tick=${{ fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" }}
                label=${{
                  value: "Wall-clock time (minutes)",
                  position: "insideBottom",
                  offset: -16,
                  style: { fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" },
                }}
              />
              <${YAxis}
                type="number"
                domain=${yDomain}
                allowDataOverflow=${true}
                tick=${{ fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" }}
                label=${{
                  value: "Best score",
                  angle: -90,
                  position: "insideLeft",
                  offset: 16,
                  style: { fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" },
                }}
              />
              <${Tooltip} content=${customTooltip} />
              <${Legend}
                verticalAlign="top"
                height=${28}
                content=${renderLegend}
              />
              ${lines.map(function (l) {
                return html`
                  <${Area}
                    key=${l.key + "_lo"}
                    type="monotone"
                    dataKey=${l.lo}
                    stackId=${l.key}
                    stroke="none"
                    fill="transparent"
                    legendType="none"
                    connectNulls=${false}
                    isAnimationActive=${false}
                    dot=${false}
                  />
                  <${Area}
                    key=${l.key + "_band"}
                    type="monotone"
                    dataKey=${l.band}
                    stackId=${l.key}
                    stroke="none"
                    fill=${l.color}
                    fillOpacity=${0.18}
                    legendType="none"
                    connectNulls=${false}
                    isAnimationActive=${false}
                    dot=${false}
                  />
                  <${Line}
                    key=${l.key}
                    type="monotone"
                    dataKey=${l.key}
                    stroke=${l.color}
                    strokeWidth=${2.2}
                    strokeDasharray=${l.dashed ? "6 3" : "0"}
                    dot=${{ r: 3 }}
                    name=${l.label}
                    connectNulls=${false}
                    isAnimationActive=${true}
                  />
                `;
              })}
            <//>
          <//>
        </div>

        <div className="chart-readout">
          ${ksel === "K=8"
            ? "K=8 (spec) stagnates near 0.97 because the speculative prompt is built before the current iteration's children are written to the database, leaving sibling history stale. K=8 (added children) fixes this by inserting all K children into the speculation prompt before evaluation finishes — it converges to 0.999, matching the standard run, by 105 min. Bands are mean ± 1σ over 3 seeds."
            : "At K=4 speculation is a mild win — slightly faster convergence with no quality cost. Bands are mean ± 1σ over 3 seeds."
          }
        </div>
      </div>
    `;
  }

  // ---- Speculation hit timeline ----
  // Per-iteration scatter: green dot = spec-hit, red dot = spec-miss.
  // Right axis shows the running cumulative hit rate.
  function SpecHitTimeline() {
    var ks = useState("K=4");
    var k = ks[0], setK = ks[1];
    var rs = useState("run1");
    var run = rs[0], setRun = rs[1];

    var data = useMemo(function () {
      var rows = ((SPEC_HIT_DATA[k] || {})[run]) || [];
      return rows.map(function (r) {
        return {
          i: r.i,
          llm: r.llm,
          hit_llm: r.hit === true  ? r.llm : null,
          miss_llm: r.hit === false ? r.llm : null,
          best: r.best,
          hit: r.hit,
        };
      });
    }, [k, run]);

    var summary = useMemo(function () {
      var rows = ((SPEC_HIT_DATA[k] || {})[run]) || [];
      var h = rows.filter(function (r) { return r.hit === true; });
      var m = rows.filter(function (r) { return r.hit === false; });
      var avgH = h.length ? (h.reduce(function (a, r) { return a + r.llm; }, 0) / h.length).toFixed(1) : "—";
      var avgM = m.length ? (m.reduce(function (a, r) { return a + r.llm; }, 0) / m.length).toFixed(1) : "—";
      return { hits: h.length, misses: m.length, avgH: avgH, avgM: avgM };
    }, [k, run]);

    function customTooltip(props) {
      if (!props.active || !props.payload || !props.payload.length) return null;
      var row = data.find(function (d) { return d.i === props.label; });
      if (!row) return null;
      var hitColor = row.hit ? "#1f9d55" : "#c43d3d";
      var hitLabel = row.hit ? "✓ spec-hit" : "✗ spec-miss";
      return html`
        <div style=${{ fontFamily: "var(--sans)", fontSize: 12, background: "white", border: "1px solid #ddd", padding: "8px 10px", borderRadius: 4, lineHeight: 1.6 }}>
          <div style=${{ marginBottom: 4, fontWeight: 600 }}>Iteration ${row.i}</div>
          <div style=${{ color: hitColor, marginBottom: 2 }}>${hitLabel}</div>
          <div>LLM time: ${row.llm}s</div>
          <div>Best score: ${row.best.toFixed(5)}</div>
        </div>`;
    }

    return html`
      <div>
        <div className="toggle-row">
          ${["K=4", "K=8"].map(function (name) {
            return html`<button key=${name} className=${name === k ? "active" : ""} onClick=${function () { setK(name); setRun("run1"); }}>${name}</button>`;
          })}
          <span style=${{ display: "inline-block", width: 12 }}></span>
          ${["run1", "run2", "run3"].map(function (name) {
            return html`<button key=${name} className=${name === run ? "active" : ""} onClick=${function () { setRun(name); }}>${name}</button>`;
          })}
        </div>
        <div className="chart-wrap">
          <${ResponsiveContainer} width="100%" height="100%">
            <${ComposedChart} data=${data} margin=${{ top: 10, right: 16, left: 0, bottom: 28 }}>
              <${CartesianGrid} strokeDasharray="3 3" stroke="#eee" />
              <${XAxis}
                dataKey="i"
                type="number"
                allowDecimals=${false}
                tick=${{ fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" }}
                label=${{ value: "Iteration", position: "insideBottom", offset: -16, style: { fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" } }}
              />
              <${YAxis}
                type="number"
                tick=${{ fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" }}
                label=${{ value: "LLM gen time (s)", angle: -90, position: "insideLeft", offset: 16, style: { fontSize: 12, fill: "#504f4f", fontFamily: "var(--sans)" } }}
              />
              <${Tooltip} content=${customTooltip} />
              <${Legend}
                verticalAlign="top"
                height=${28}
                content=${renderLegend}
              />
              <${Line}
                type="monotone"
                dataKey="llm"
                stroke="#ccc"
                strokeWidth=${1.2}
                dot=${false}
                legendType="none"
                isAnimationActive=${false}
              />
              <${Line}
                dataKey="hit_llm"
                stroke="#1f9d55"
                strokeWidth=${0}
                dot=${{ r: 6, fill: "#1f9d55", strokeWidth: 0 }}
                activeDot=${{ r: 8 }}
                name="Spec hit"
                connectNulls=${false}
                isAnimationActive=${false}
              />
              <${Line}
                dataKey="miss_llm"
                stroke="#c43d3d"
                strokeWidth=${0}
                dot=${{ r: 4, fill: "#c43d3d", strokeWidth: 0 }}
                activeDot=${{ r: 6 }}
                name="Spec miss"
                connectNulls=${false}
                isAnimationActive=${false}
              />
            <//>
          <//>
        </div>
        <div className="chart-readout">
          Green dots are hit iterations, red dots are misses, and the height of each dot is the LLM generation time seen at that iteration.
          ${summary.hits} hits · ${summary.misses} misses · mean LLM time: hit ${summary.avgH}s · miss ${summary.avgM}s.
        </div>
      </div>
    `;
  }

  // ---- Mount ----
  try {
    var kRoot = document.getElementById("kslider-root");
    if (kRoot) createRoot(kRoot).render(createElement(KSlider));
  } catch (err) {
    console.error("KSlider mount failed:", err);
    showError("kslider-root", "Widget failed to mount: " + err.message);
  }

  try {
    var rRoot = document.getElementById("results-root");
    if (rRoot) createRoot(rRoot).render(createElement(ResultsChart));
  } catch (err) {
    console.error("ResultsChart mount failed:", err);
    showError("results-root", "Widget failed to mount: " + err.message);
  }

  try {
    var sRoot = document.getElementById("spec-results-root");
    if (sRoot) createRoot(sRoot).render(createElement(SpecChart));
  } catch (err) {
    console.error("SpecChart mount failed:", err);
    showError("spec-results-root", "Widget failed to mount: " + err.message);
  }

  try {
    var scRoot = document.getElementById("spec-cand-root");
    if (scRoot) createRoot(scRoot).render(createElement(SpecCandChart));
  } catch (err) {
    console.error("SpecCandChart mount failed:", err);
    showError("spec-cand-root", "Widget failed to mount: " + err.message);
  }

  try {
    var stRoot = document.getElementById("spec-hit-root");
    if (stRoot) createRoot(stRoot).render(createElement(SpecHitTimeline));
  } catch (err) {
    console.error("SpecHitTimeline mount failed:", err);
    showError("spec-hit-root", "Widget failed to mount: " + err.message);
  }
})();
