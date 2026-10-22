import { useEffect } from "react";

/** Fired on ⌘K / Ctrl+K; the mounted command bar listens and takes focus. */
export const FOCUS_COMMAND_EVENT = "papertrail:focus-command";

export function focusCommandBar(): void {
  window.dispatchEvent(new Event(FOCUS_COMMAND_EVENT));
}

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

/**
 * Global shortcuts: ⌘K / Ctrl+K or "/" focus the command bar, Escape runs
 * `onEscape` (closes the evidence drawer or the mobile sidebar).
 */
export function useHotkeys({ onEscape }: { onEscape: () => void }): void {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const mod = event.metaKey || event.ctrlKey;
      if (mod && event.key.toLowerCase() === "k") {
        event.preventDefault();
        focusCommandBar();
      } else if (event.key === "/" && !mod && !isTyping(event.target)) {
        event.preventDefault();
        focusCommandBar();
      } else if (event.key === "Escape") {
        onEscape();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onEscape]);
}
