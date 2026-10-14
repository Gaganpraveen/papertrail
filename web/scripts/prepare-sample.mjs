import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
const file = new URL("../public/samples/active-retrieval.pdf", import.meta.url);
const expected =
  "5536c9136a510c9430a02a3424963ef7e3c2338e8e421ff51c98e7db3480d34c";
let bytes;
try {
  bytes = await readFile(file);
} catch {
  const response = await fetch(
    "https://aclanthology.org/2023.emnlp-main.495.pdf",
    { signal: AbortSignal.timeout(30000) },
  );
  if (!response.ok)
    throw Error(`Sample PDF download failed (${response.status}).`);
  bytes = Buffer.from(await response.arrayBuffer());
}
if (createHash("sha256").update(bytes).digest("hex") !== expected)
  throw Error(
    "Sample PDF checksum changed; review the upstream document before updating the pin.",
  );
await mkdir(new URL("../public/samples/", import.meta.url), {
  recursive: true,
});
await writeFile(file, bytes);
console.log("Verified openly licensed sample PDF.");
