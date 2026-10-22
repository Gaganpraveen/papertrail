import { ChevronRight, Menu } from "lucide-react";
import type { View } from "@/store/types";
import { IconButton } from "@/ui/kit";
import { store, useStore } from "@/ui/hooks/store";
import { BrandMark } from "./Brand";
import { CommandBar } from "./CommandBar";

function useCrumb(view: View): string {
  const paperTitle = useStore((s) => (view.kind === "paper" ? s.papers[view.id]?.meta.title : undefined));
  switch (view.kind) {
    case "home":
      return "New analysis";
    case "search":
      return "Topic search";
    case "paper":
      return paperTitle ?? "Paper";
    case "survey":
      return "Literature survey";
  }
}

export function TopBar({ onMenu }: { onMenu?: () => void }) {
  const view = useStore((s) => s.view);
  const crumb = useCrumb(view);
  const home = view.kind === "home";

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-line bg-bg/80 px-4 backdrop-blur">
      {onMenu && (
        <>
          <IconButton label="Open menu" onClick={onMenu} className="-ml-1">
            <Menu className="size-4" />
          </IconButton>
          <button onClick={() => store().setView({ kind: "home" })} aria-label="Home">
            <BrandMark className="size-6" />
          </button>
        </>
      )}
      <nav aria-label="Breadcrumb" className="hidden min-w-0 items-center gap-1.5 text-xs text-ink-3 lg:flex lg:max-w-[34%]">
        <button onClick={() => store().setView({ kind: "home" })} className="hover:text-ink">
          PaperTrail
        </button>
        <ChevronRight className="size-3.5 shrink-0" />
        <span className="truncate text-ink-2">{crumb}</span>
      </nav>
      <div className="ml-auto w-full max-w-xl min-w-0">{!home && <CommandBar />}</div>
    </header>
  );
}
