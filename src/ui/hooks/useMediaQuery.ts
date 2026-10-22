import { useSyncExternalStore } from "react";

/** Live boolean for a CSS media query (false during non-browser rendering). */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

/** The sidebar turns into a drawer below this width. */
export const COMPACT_LAYOUT_QUERY = "(max-width: 899px)";
