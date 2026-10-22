/**
 * Application shell: sidebar, top command bar, the active view, the evidence
 * drawer and the live trace dock. Views never talk to the pipeline directly;
 * they read the library store and call its actions.
 */
import { useCallback, useEffect, useState } from "react";
import { cn } from "@/ui/kit";
import { store, useStore } from "@/ui/hooks/store";
import { useFileDrop } from "@/ui/hooks/useFileDrop";
import { useHashRoute } from "@/ui/hooks/useHashRoute";
import { useHotkeys } from "@/ui/hooks/useHotkeys";
import { COMPACT_LAYOUT_QUERY, useMediaQuery } from "@/ui/hooks/useMediaQuery";
import { useSurveyIntentRunner } from "@/ui/hooks/usePendingSurvey";
import { EvidenceDrawer } from "@/ui/evidence/EvidenceDrawer";
import { DropOverlay } from "@/ui/layout/DropOverlay";
import { Sidebar } from "@/ui/layout/Sidebar";
import { SurveyIntentBanner } from "@/ui/layout/SurveyIntentBanner";
import { Toasts } from "@/ui/layout/Toasts";
import { TopBar } from "@/ui/layout/TopBar";
import { TraceDock } from "@/ui/trace/TraceDock";
import HomeView from "@/ui/views/home/HomeView";
import PaperView from "@/ui/views/paper";
import SearchView from "@/ui/views/search/SearchView";
import SurveyView from "@/ui/views/survey/SurveyView";

let hydrationStarted = false;

function ViewRouter() {
  const view = useStore((s) => s.view);
  switch (view.kind) {
    case "home":
      return <HomeView />;
    case "search":
      return <SearchView />;
    case "paper":
      return <PaperView key={view.id} id={view.id} />;
    case "survey":
      return <SurveyView key={view.id} id={view.id} />;
  }
}

function viewKey(view: ReturnType<typeof store>["view"]): string {
  return view.kind === "paper" || view.kind === "survey" ? `${view.kind}:${view.id}` : view.kind;
}

export default function App() {
  const compact = useMediaQuery(COMPACT_LAYOUT_QUERY);
  const [navOpen, setNavOpen] = useState(false);
  const view = useStore((s) => s.view);

  useEffect(() => {
    if (hydrationStarted) return; // StrictMode mounts twice in development
    hydrationStarted = true;
    void store().hydrate();
  }, []);

  useHashRoute();
  useSurveyIntentRunner();

  const onFiles = useCallback((files: File[], rejected: number) => {
    const { submitFiles, toast } = store();
    if (rejected > 0) toast(`${rejected} file(s) skipped: only PDFs can be analysed.`, "warn");
    if (files.length > 0) void submitFiles(files);
  }, []);
  const dragging = useFileDrop(onFiles);

  const onEscape = useCallback(() => {
    const { evidence, openEvidence } = store();
    if (evidence) openEvidence(null);
    else setNavOpen(false);
  }, []);
  useHotkeys({ onEscape });

  return (
    <div className="flex h-full overflow-hidden bg-bg text-ink">
      {!compact && <Sidebar />}
      {compact && (
        <>
          <div
            onClick={() => setNavOpen(false)}
            className={cn(
              "fixed inset-0 z-40 bg-bg/60 backdrop-blur-sm transition-opacity",
              navOpen ? "opacity-100" : "pointer-events-none opacity-0",
            )}
          />
          <div
            className={cn(
              "fixed inset-y-0 left-0 z-50 shadow-card transition-transform duration-200",
              navOpen ? "translate-x-0" : "-translate-x-full",
            )}
          >
            <Sidebar onNavigate={() => setNavOpen(false)} onClose={() => setNavOpen(false)} />
          </div>
        </>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onMenu={compact ? () => setNavOpen(true) : undefined} />
        <SurveyIntentBanner />
        <main key={viewKey(view)} className="animate-in min-h-0 flex-1 overflow-y-auto">
          <ViewRouter />
        </main>
        <TraceDock />
      </div>

      <EvidenceDrawer />
      <Toasts />
      {dragging && <DropOverlay />}
    </div>
  );
}
