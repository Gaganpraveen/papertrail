/**
 * Typed access to the library store. Components select small slices so a
 * change to one paper does not re-render unrelated parts of the shell.
 */
import { useLibrary } from "@/store/library";
import type { LibraryState } from "@/store/types";

export function useStore<T>(selector: (state: LibraryState) => T): T {
  return useLibrary(selector);
}

/** Non-reactive access for event handlers (actions never change identity). */
export function store(): LibraryState {
  return useLibrary.getState();
}
