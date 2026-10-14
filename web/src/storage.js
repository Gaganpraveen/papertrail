const DB = "papertrail-browser-v1";
async function database() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () =>
      r.result.createObjectStore("documents", { keyPath: "id" });
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function transaction(mode, fn) {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("documents", mode);
    const r = fn(tx.objectStore("documents"));
    let result;
    r.onsuccess = () => {
      result = r.result;
    };
    tx.oncomplete = () => {
      db.close();
      resolve(result);
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error);
    };
  });
}
export const loadDocuments = () => transaction("readonly", (s) => s.getAll());
export const saveDocument = (doc) =>
  transaction("readwrite", (s) => s.put(doc));
export const removeDocument = (id) =>
  transaction("readwrite", (s) => s.delete(id));
