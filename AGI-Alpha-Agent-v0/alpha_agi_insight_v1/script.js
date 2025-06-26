// Fetch the forecast and population data, then initialize charts
Promise.all([
  fetch('forecast.json').then(res => res.json()),
  fetch('population.json').then(res => res.json())
]).then(([forecastData, popData]) => {
  // Extract data
  const years = forecastData.years;
  const capability = forecastData.capability;
  const sectors = forecastData.sectors;
  const solutions = popData.solutions;

  // 1. Plot AGI Capability Growth Curve
  const capTrace = {
    x: years,
    y: capability,
    mode: 'lines+markers',
    name: 'AGI Capability',
    line: { color: '#d62728', width: 3 },  // red line for emphasis
    marker: { size: 6, symbol: 'circle', color: '#d62728' }
  };
  const capLayout = {
    margin: { t: 30, r: 20, l: 40, b: 40 },
    xaxis: { title: 'Year', tickmode: 'array', tickvals: years },
    yaxis: { title: 'AGI Capability (T_AGI)', rangemode: 'tozero' },
    title: { text: 'AGI Capability vs Time', font: { size: 16 } }
  };
  Plotly.newPlot('capability-chart', [capTrace], capLayout, { displayModeBar: false, responsive: true });

  // 2. Plot Sector Disruption Timeline (multi-line chart)
  const timelineTraces = sectors.map(sector => {
    // Determine marker symbols for each year: star at disruption year, circle otherwise
    const symbols = sector.values.map((val, idx) =>
      years[idx] === sector.disruptionYear ? 'star' : 'circle'
    );
    const sizes = sector.values.map((val, idx) =>
      years[idx] === sector.disruptionYear ? 12 : 6
    );
    return {
      x: years,
      y: sector.values,
      mode: 'lines+markers',
      name: sector.name,
      line: { width: 2 },  // use Plotly default color cycle
      marker: {
        size: sizes,
        symbol: symbols,
        line: { width: 1, color: '#000' }  // outline markers in black for visibility
      }
    };
  });
  const timelineLayout = {
    margin: { t: 30, r: 20, l: 40, b: 40 },
    xaxis: { title: 'Year', tickmode: 'array', tickvals: years },
    yaxis: { title: 'Sector Performance Index', rangemode: 'tozero' },
    title: { text: 'Sector Performance and Disruption Jumps', font: { size: 16 } },
    legend: { orientation: 'h', x: 0, y: -0.2 }
  };
  Plotly.newPlot('timeline-chart', timelineTraces, timelineLayout, { displayModeBar: false, responsive: true });

  // 3. Plot Pareto Frontier of Evolved Solutions
  // Separate frontier vs non-frontier points for styling
  const frontierPoints = solutions.filter(s => s.frontier);
  const otherPoints = solutions.filter(s => !s.frontier);
  const traceOthers = {
    x: otherPoints.map(p => p.time),
    y: otherPoints.map(p => p.value),
    mode: 'markers',
    name: 'Other Solutions',
    marker: { color: 'rgba(100,100,100,0.5)', size: 8, symbol: 'circle' },
    hovertemplate: 'Time: %{x} yr<br>Value: %{y} trillion<extra></extra>'
  };
  const traceFrontier = {
    x: frontierPoints.map(p => p.time),
    y: frontierPoints.map(p => p.value),
    mode: 'markers+lines',
    name: 'Pareto Frontier',
    marker: { color: '#1f77b4', size: 10, symbol: 'diamond' },
    line: { color: '#1f77b4', dash: 'solid', width: 2 },
    hovertemplate: 'Time: %{x} yr<br>Value: %{y} trillion<extra></extra>'
  };
  const paretoLayout = {
    margin: { t: 30, r: 20, l: 50, b: 50 },
    xaxis: { title: 'Time to Disruption (years)', dtick: 1, range: [0.5, 5.5] },
    yaxis: { title: 'Economic Value (USD trillions)', rangemode: 'tozero' },
    title: { text: 'Evolved Solutions Trade-off (Value vs Time)', font: { size: 16 } },
    legend: { x: 0.02, y: 0.98 }
  };
  Plotly.newPlot('pareto-chart', [traceOthers, traceFrontier], paretoLayout, { displayModeBar: false, responsive: true });

  // 4. Populate Agent Logs
  const logsElement = document.getElementById('logs-panel');
  const logLines = [
    "[PlanningAgent] Initializing high-level plan and setting 5-year insight horizon.",
    "[ResearchAgent] Gathering domain data for all sectors (offline knowledge base).",
    "[StrategyAgent] Scoring sectors by AGI disruption risk…",
    "[StrategyAgent] -> Top sector identified: Transportation (imminent AGI impact).",
    "[MarketAgent] Estimating economic upside for Transportation: ~$1.5 trillion in first year.",
    "[CodeGenAgent] Generating prototype AGI solutions for Transportation sector.",
    "[SafetyGuardianAgent] Reviewing proposed strategies for alignment with safety policies.",
    "[PlanningAgent] Plan updated. Next target sector: Finance (year 2).",
    "[MemoryAgent] Logging outcome of year 1 disruption (Transportation) to ledger.",
    "----",
    "[PlanningAgent] Proceeding to next iteration with refined strategies…"
  ];
  logsElement.textContent = logLines.join("\n");
}).catch(err => {
  console.error("Error loading data files:", err);
});

// Toggle the visibility of the logs panel
const toggleBtn = document.getElementById('toggle-logs');
toggleBtn.addEventListener('click', () => {
  const panel = document.getElementById('logs-panel');
  panel.classList.toggle('hidden');
  const expanded = !panel.classList.contains('hidden');
  toggleBtn.setAttribute('aria-expanded', expanded.toString());
});
