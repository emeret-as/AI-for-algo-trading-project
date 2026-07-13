import { useEffect, useMemo, useState } from 'react';

function App() {
  const [form, setForm] = useState({
    date: '2026-04-06',
    underlying: 'SPY',
    spot: '450',
    regime: 'normal',
    seed: '42',
    num_strikes: '21',
    num_monthly: '4',
    include_weeklies: true,
    dry_run: true,
  });
  const [replayForm, setReplayForm] = useState({
    start_date: '2026-04-01',
    end_date: '2026-04-06',
    underlying: 'SPY',
    spot: '450',
    regime: 'normal',
    seed: '42',
    num_strikes: '21',
    num_monthly: '4',
    include_weeklies: true,
  });
  const [loading, setLoading] = useState(false);
  const [replayLoading, setReplayLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [replayResult, setReplayResult] = useState(null);
  const [configText, setConfigText] = useState('');
  const [error, setError] = useState('');
  const [replayError, setReplayError] = useState('');

  const metricCards = useMemo(() => {
    if (!result) return [];
    const qc = result.qc_report?.summary || {};
    return [
      { label: 'Universe', value: result.universe_size ?? '—' },
      { label: 'Raw events', value: result.raw_events ?? '—' },
      { label: 'QC pass rate', value: `${Math.round(qc.pass_rate ?? 0)}%` },
      { label: 'IV solved', value: `${result.iv_converged ?? 0}/${result.iv_total ?? 0}` },
      { label: 'Surface slices', value: result.surface_slices ?? '—' },
      { label: 'Risk positions', value: result.risk_positions ?? '—' },
    ];
  }, [result]);

  const chartData = useMemo(() => {
    if (!result) return [];
    const qc = result.qc_report?.summary || {};
    return [
      { label: 'QC pass', value: qc.pass ?? 0, max: Math.max(1, qc.total_checks ?? 1) },
      { label: 'QC warns', value: qc.warn ?? 0, max: Math.max(1, qc.total_checks ?? 1) },
      { label: 'QC fails', value: qc.fail ?? 0, max: Math.max(1, qc.total_checks ?? 1) },
      { label: 'IV converged', value: result.iv_converged ?? 0, max: Math.max(1, result.iv_total ?? 1) },
    ];
  }, [result]);

  useEffect(() => {
    const loadConfig = async () => {
      try {
        const response = await fetch('/api/config');
        const data = await response.json();
        setConfigText(data.config || '');
      } catch (err) {
        setConfigText('Unable to load configuration snapshot.');
      }
    };
    loadConfig();
  }, []);

  const runPipeline = async () => {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('/api/run-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          date: form.date,
          underlying: form.underlying,
          spot: Number(form.spot),
          regime: form.regime,
          seed: Number(form.seed),
          num_strikes: Number(form.num_strikes),
          num_monthly: Number(form.num_monthly),
          include_weeklies: form.include_weeklies,
          dry_run: form.dry_run,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || data.error || 'Pipeline failed');
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runReplay = async () => {
    setReplayLoading(true);
    setReplayError('');
    setReplayResult(null);

    try {
      const response = await fetch('/api/replay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_date: replayForm.start_date,
          end_date: replayForm.end_date,
          underlying: replayForm.underlying,
          spot: Number(replayForm.spot),
          regime: replayForm.regime,
          seed: Number(replayForm.seed),
          num_strikes: Number(replayForm.num_strikes),
          num_monthly: Number(replayForm.num_monthly),
          include_weeklies: replayForm.include_weeklies,
          dry_run: true,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Replay failed');
      }
      setReplayResult(data);
    } catch (err) {
      setReplayError(err.message);
    } finally {
      setReplayLoading(false);
    }
  };

  const updateField = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const updateReplayField = (field, value) => {
    setReplayForm((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Options analytics workspace</p>
          <h1>Volatility Pipeline Dashboard</h1>
          <p className="hero-copy">Run the Python options pipeline from a polished React interface and inspect QC and risk outcomes.</p>
        </div>
        <button onClick={runPipeline} disabled={loading}>{loading ? 'Running…' : 'Run pipeline'}</button>
      </header>

      <section className="panel form-panel">
        <div className="section-title">
          <h2>Pipeline parameters</h2>
          <span>Fine-tune the synthetic market run</span>
        </div>

        <div className="form-grid">
          <label>
            Trade date
            <input value={form.date} onChange={(e) => updateField('date', e.target.value)} />
          </label>
          <label>
            Underlying
            <input value={form.underlying} onChange={(e) => updateField('underlying', e.target.value)} />
          </label>
          <label>
            Spot price
            <input type="number" value={form.spot} onChange={(e) => updateField('spot', e.target.value)} />
          </label>
          <label>
            Regime
            <select value={form.regime} onChange={(e) => updateField('regime', e.target.value)}>
              <option value="calm">Calm</option>
              <option value="normal">Normal</option>
              <option value="stressed">Stressed</option>
              <option value="sparse">Sparse</option>
            </select>
          </label>
          <label>
            Seed
            <input type="number" value={form.seed} onChange={(e) => updateField('seed', e.target.value)} />
          </label>
          <label>
            Num strikes
            <input type="number" value={form.num_strikes} onChange={(e) => updateField('num_strikes', e.target.value)} />
          </label>
          <label>
            Num monthly
            <input type="number" value={form.num_monthly} onChange={(e) => updateField('num_monthly', e.target.value)} />
          </label>
          <label className="checkbox-row">
            <input type="checkbox" checked={form.include_weeklies} onChange={(e) => updateField('include_weeklies', e.target.checked)} />
            <span>Include weeklies</span>
          </label>
          <label className="checkbox-row">
            <input type="checkbox" checked={form.dry_run} onChange={(e) => updateField('dry_run', e.target.checked)} />
            <span>Dry run</span>
          </label>
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <section className="metric-grid">
            {metricCards.map((item) => (
              <div key={item.label} className="panel metric-card">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </section>

          <section className="panel chart-panel">
            <div className="section-title">
              <h2>Pipeline health</h2>
              <span>Key outcome indicators</span>
            </div>
            <div className="chart-list">
              {chartData.map((item) => (
                <div key={item.label} className="chart-row">
                  <div className="chart-labels">
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                  </div>
                  <div className="bar-track" aria-hidden="true">
                    <div className="bar-fill" style={{ width: `${Math.max(6, (item.value / item.max) * 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="panel tables-panel">
            <div className="section-title">
              <h2>Scenario outcomes</h2>
              <span>Stress and regime results</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Scenario</th>
                    <th>Portfolio</th>
                    <th>Total PnL</th>
                    <th>Worst contributor</th>
                    <th>Worst PnL</th>
                    <th>Positions</th>
                  </tr>
                </thead>
                <tbody>
                  {(result.scenario_table || []).map((row) => (
                    <tr key={row.scenario}>
                      <td>{row.scenario}</td>
                      <td>{row.portfolio}</td>
                      <td>{row.total_pnl}</td>
                      <td>{row.worst_contributor}</td>
                      <td>{row.worst_pnl}</td>
                      <td>{row.positions}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel tables-panel">
            <div className="section-title">
              <h2>QC checks</h2>
              <span>Per-check outcomes</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Check</th>
                    <th>Target</th>
                    <th>Status</th>
                    <th>Measured</th>
                    <th>Threshold</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {(result.qc_table || []).map((row) => (
                    <tr key={`${row.check}-${row.target}`}>
                      <td>{row.check}</td>
                      <td>{row.target}</td>
                      <td><span className={`status-pill ${row.status}`}>{row.status}</span></td>
                      <td>{Number(row.measured).toFixed(3)}</td>
                      <td>{Number(row.threshold).toFixed(3)}</td>
                      <td>{row.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel tables-panel">
            <div className="section-title">
              <h2>Risk metrics</h2>
              <span>Portfolio-level aggregates</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Group</th>
                    <th>Portfolio</th>
                    <th>Positions</th>
                    <th>Delta</th>
                    <th>Gamma</th>
                    <th>Vega</th>
                    <th>Theta</th>
                    <th>$ Gamma</th>
                    <th>$ Vega</th>
                  </tr>
                </thead>
                <tbody>
                  {(result.risk_table || []).map((row) => (
                    <tr key={`${row.group}-${row.portfolio}`}>
                      <td>{row.group}</td>
                      <td>{row.portfolio}</td>
                      <td>{row.positions}</td>
                      <td>{row.delta}</td>
                      <td>{row.gamma}</td>
                      <td>{row.vega}</td>
                      <td>{row.theta}</td>
                      <td>{row['$ gamma']}</td>
                      <td>{row['$ vega']}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      <section className="panel form-panel">
        <div className="section-title">
          <h2>Historical replay</h2>
          <span>Re-run the pipeline over a date range</span>
        </div>
        <div className="form-grid">
          <label>
            Start date
            <input value={replayForm.start_date} onChange={(e) => updateReplayField('start_date', e.target.value)} />
          </label>
          <label>
            End date
            <input value={replayForm.end_date} onChange={(e) => updateReplayField('end_date', e.target.value)} />
          </label>
          <label>
            Underlying
            <input value={replayForm.underlying} onChange={(e) => updateReplayField('underlying', e.target.value)} />
          </label>
          <label>
            Spot price
            <input type="number" value={replayForm.spot} onChange={(e) => updateReplayField('spot', e.target.value)} />
          </label>
          <label>
            Regime
            <select value={replayForm.regime} onChange={(e) => updateReplayField('regime', e.target.value)}>
              <option value="calm">Calm</option>
              <option value="normal">Normal</option>
              <option value="stressed">Stressed</option>
              <option value="sparse">Sparse</option>
            </select>
          </label>
          <label>
            Seed
            <input type="number" value={replayForm.seed} onChange={(e) => updateReplayField('seed', e.target.value)} />
          </label>
          <label className="checkbox-row">
            <input type="checkbox" checked={replayForm.include_weeklies} onChange={(e) => updateReplayField('include_weeklies', e.target.checked)} />
            <span>Include weeklies</span>
          </label>
        </div>
        <button onClick={runReplay} disabled={replayLoading} style={{ marginTop: '0.9rem' }}>
          {replayLoading ? 'Replaying…' : 'Run replay'}
        </button>
        {replayError && <div className="error">{replayError}</div>}
        {replayResult && (
          <div className="table-wrap" style={{ marginTop: '0.9rem' }}>
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Status</th>
                  <th>Universe</th>
                  <th>QC pass rate</th>
                  <th>Worst scenario</th>
                </tr>
              </thead>
              <tbody>
                {(replayResult.results || []).map((row) => (
                  <tr key={row.trade_date || row.replay_date || row.error}>
                    <td>{row.trade_date || row.replay_date || '—'}</td>
                    <td>{row.error ? 'error' : 'ok'}</td>
                    <td>{row.universe_size ?? '—'}</td>
                    <td>{row.qc_report?.summary?.pass_rate ? `${Math.round(row.qc_report.summary.pass_rate)}%` : '—'}</td>
                    <td>{row.worst_scenario || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel form-panel">
        <div className="section-title">
          <h2>Configuration snapshot</h2>
          <span>The defaults loaded by the dashboard</span>
        </div>
        <pre className="config-box">{configText}</pre>
      </section>
    </div>
  );
}

export default App;
