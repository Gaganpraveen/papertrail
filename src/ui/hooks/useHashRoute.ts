/**
 * Keeps the store's view in sync with the URL hash (see routes.ts), so the
 * back button works and views can be linked to.
 */
import { useEffect, useRef } from "react";
import type { LibraryState, View } from "@/store/types";
import { hashToRoute, viewToHash } from "./routes";
import { store, useStore } from "./store";

/** Only navigate to things that exist in the library. */
function resolvable(route: View, state: LibraryState): boolean {
  if (route.kind === "paper") return route.id in state.papers;
  if (route.kind === "survey") return route.id in state.surveys;
  if (route.kind === "search") return state.search !== null;
  return true;
}

function applyHash(hash: string): void {
  const route = hashToRoute(hash);
  const state = store();
  if (!route) return;
  if (route.kind === "analyze") {
    window.history.replaceState(null, "", "#/");
    void state.submit(route.input);
    return;
  }
  const target = resolvable(route, state) ? route : ({ kind: "home" } as const);
  if (viewToHash(target) !== viewToHash(state.view)) state.setView(target);
}

export function useHashRoute(): void {
  const hydrated = useStore((s) => s.hydrated);
  const view = useStore((s) => s.view);
  const synced = useRef(false);

  // Initial deep link, once the library has been restored from IndexedDB.
  // A view the user opened while it was loading wins over the URL.
  useEffect(() => {
    if (hydrated && store().view.kind === "home") applyHash(window.location.hash);
  }, [hydrated]);

  // View -> URL. Reads the live view: the deep link above may have changed it
  // in this same commit, and the rendered `view` would still be the old one.
  useEffect(() => {
    if (!hydrated) return;
    const next = viewToHash(store().view);
    const first = !synced.current;
    synced.current = true;
    if (next === window.location.hash) return;
    // The first write replaces the URL the page was opened with, so Back leaves instead of landing on it.
    if (first || window.location.hash === "" || window.location.hash === "#") {
      window.history.replaceState(null, "", next);
    } else {
      window.history.pushState(null, "", next);
    }
  }, [view, hydrated]);

  // URL -> view (back / forward).
  useEffect(() => {
    const onPop = () => applyHash(window.location.hash);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
}
