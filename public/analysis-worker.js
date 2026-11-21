/* global importScripts, loadPyodide */
let enginePromise;

function loadSearch() {
  if (!enginePromise) {
    enginePromise = (async () => {
      importScripts("https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js");
      const python = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.0/full/",
      });
      const sources = await Promise.all(
        ["engine.py", "search.py"].map(async (name) => {
          const response = await fetch(`/${name}`, { cache: "no-cache" });
          if (!response.ok)
            throw new Error(`Could not load ${name} (${response.status})`);
          return [name, await response.text()];
        })
      );
      for (const [name, source] of sources) python.FS.writeFile(name, source);
      return python.pyimport("search");
    })();
  }
  return enginePromise;
}

self.onmessage = async ({ data: { id, state, options } }) => {
  try {
    const search = await loadSearch();
    const progress = (json) => {
      self.postMessage({ id, type: "progress", result: JSON.parse(json) });
    };
    const json = search.analyze_json(
      state,
      options.timeMs,
      options.maxDepth,
      options.beamWidth,
      options.multiPv,
      progress
    );
    self.postMessage({ id, type: "result", result: JSON.parse(json) });
  } catch (error) {
    self.postMessage({
      id,
      type: "error",
      error: String(error?.message || error),
    });
  }
};
