/**
 * Content hashing for uploads. A SHA-256 of the PDF bytes gives every uploaded
 * file a stable identity ("upload:<first 12 hex chars>"), so re-uploading the
 * same paper reuses its cached analysis instead of creating a duplicate.
 * Uses the browser's native Web Crypto (also available in Node 20+).
 */

export async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("Web Crypto is unavailable (a secure https or localhost context is required)");
  const digest = await subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
