import { useEffect, useState } from 'react';
import { analysisApi, graphApi } from '@/api/client';
import type {
  HealthScore as HealthScoreType,
  DeadCodeReport as DeadCodeReportType,
  CouplingPair,
  Community,
  Process,
  OverviewStats,
} from '@/types';
import { HealthScore } from './HealthScore';
import { QuickStats } from './QuickStats';
import { DeadCodeReport } from './DeadCodeReport';
import { CouplingHeatmap } from './CouplingHeatmap';
import { InheritanceTree } from './InheritanceTree';
import { BranchDiff } from './BranchDiff';
import { LoadingSpinner } from '@/components/shared/LoadingSpinner';

interface DashboardData {
  health: HealthScoreType | null;
  deadCode: DeadCodeReportType | null;
  coupling: CouplingPair[];
  communities: Community[];
  processes: Process[];
  overview: OverviewStats | null;
}

const CARD_STYLE: React.CSSProperties = {
  background: 'var(--bg-surface)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius)',
  overflow: 'hidden',
  display: 'flex',
  flexDirection: 'column',
};

const HEADING_STYLE: React.CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
  fontSize: 13,
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  color: 'var(--text-bright)',
  padding: '6px 8px',
  borderBottom: '1px solid var(--border)',
  margin: 0,
  flexShrink: 0,
};

function Card({
  title,
  children,
  style,
  loading: isLoading,
}: {
  title: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
  loading?: boolean;
}) {
  return (
    <div style={{ ...CARD_STYLE, ...style, position: 'relative' }}>
      <h3 style={HEADING_STYLE}>{title}</h3>
      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        {isLoading ? <LoadingSpinner /> : children}
      </div>
    </div>
  );
}

export function AnalysisView() {
  const [data, setData] = useState<DashboardData>({
    health: null,
    deadCode: null,
    coupling: [],
    communities: [],
    processes: [],
    overview: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([
      analysisApi.getHealth().catch(() => null),
      analysisApi.getDeadCode().catch(() => null),
      analysisApi.getCoupling().catch(() => ({ pairs: [] as CouplingPair[] })),
      analysisApi.getCommunities().catch(() => ({ communities: [] as Community[] })),
      analysisApi.getProcesses().catch(() => ({ processes: [] as Process[] })),
      graphApi.getOverview().catch(() => null),
    ])
      .then(([health, deadCode, couplingResp, commResp, procResp, overview]) => {
        if (cancelled) return;
        setData({
          health: health ?? null,
          deadCode: deadCode ?? null,
          coupling: couplingResp?.pairs ?? [],
          communities: commResp?.communities ?? [],
          processes: procResp?.processes ?? [],
          overview: overview ?? null,
        });
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div
        style={{
          padding: 16,
          color: 'var(--danger)',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12,
        }}
      >
        Failed to load analysis data: {error}
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 2fr',
        gridTemplateRows: 'auto auto auto',
        gap: 8,
        padding: 8,
        height: '100%',
        overflow: 'auto',
        opacity: 1,
      }}
    >
      {/* Row 1 */}
      <Card title="Health Score" loading={loading}>
        <HealthScore data={data.health} />
      </Card>
      <Card title="Quick Stats" loading={loading}>
        <QuickStats
          overview={data.overview}
          health={data.health}
          deadCode={data.deadCode}
          coupling={data.coupling}
          communities={data.communities}
          processes={data.processes}
        />
      </Card>

      {/* Row 2 */}
      <Card title="Dead Code Report" style={{ gridColumn: 'span 1' }} loading={loading}>
        <DeadCodeReport data={data.deadCode} />
      </Card>
      <Card title="Coupling Heatmap" style={{ gridColumn: 'span 1' }} loading={loading}>
        <CouplingHeatmap pairs={data.coupling} />
      </Card>

      {/* Row 3 */}
      <Card title="Inheritance Tree" style={{ gridColumn: 'span 1' }} loading={loading}>
        <InheritanceTree />
      </Card>
      <Card title="Branch Diff" style={{ gridColumn: 'span 1' }} loading={loading}>
        <BranchDiff />
      </Card>
    </div>
  );
}
