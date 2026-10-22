import { useCallback, useState } from "react";

/**
 * useState that remembers its value in localStorage. Storage can be
 * unavailable (private mode, blocked site data), so every access is guarded
 * and the hook silently falls back to in-memory state.
 */
export function usePersistentState<T>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? initial : (JSON.parse(raw) as T);
    } catch {
      return initial;
    }
  });

  const update = useCallback(
    (next: T) => {
      setValue(next);
      try {
        localStorage.setItem(key, JSON.stringify(next));
      } catch {
        // Persistence is a convenience; the in-memory value is still correct.
      }
    },
    [key],
  );

  return [value, update];
}
