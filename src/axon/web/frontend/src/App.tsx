import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { PanelLayout } from '@/components/layout/PanelLayout';
import { ExplorerSidebar } from '@/components/explorer/ExplorerSidebar';
import { useViewStore } from '@/stores/viewStore';

export function App() {
  const activeView = useViewStore((s) => s.activeView);

  return (
    <div
      className="h-screen w-screen flex flex-col overflow-hidden"
      style={{ background: 'var(--bg-primary)' }}
    >
      <Header />
      <main className="flex-1 overflow-hidden">
        {activeView === 'explorer' && (
          <PanelLayout
            left={<ExplorerSidebar />}
            center={
              <div
                className="flex items-center justify-center h-full"
                style={{ color: 'var(--text-dimmed)' }}
              >
                Graph Canvas
              </div>
            }
            right={
              <div className="p-2" style={{ color: 'var(--text-secondary)' }}>
                Detail Panel
              </div>
            }
          />
        )}
        {activeView === 'analysis' && (
          <div className="p-4" style={{ color: 'var(--text-secondary)' }}>
            Analysis Dashboard
          </div>
        )}
        {activeView === 'cypher' && (
          <div className="p-4" style={{ color: 'var(--text-secondary)' }}>
            Cypher Console
          </div>
        )}
      </main>
      <StatusBar />
    </div>
  );
}
