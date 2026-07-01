import type { GraphManifestItem, PatientGraph } from '../types/graph';

const shardCache = new Map<string, Promise<PatientGraph[]>>();

export async function loadManifest(): Promise<GraphManifestItem[]> {
  const response = await fetch('/graphs/manifest.json');
  if (!response.ok) {
    throw new Error('Could not load graph manifest.');
  }
  return response.json();
}

export async function loadGraph(file: string): Promise<PatientGraph> {
  const [path, offset] = file.split('::');
  if (offset !== undefined) {
    if (!shardCache.has(path)) {
      shardCache.set(
        path,
        fetch(`/graphs/${path}`).then(async (response) => {
          if (!response.ok) throw new Error(`Could not load graph shard: ${path}`);
          const payload = await response.json();
          return Array.isArray(payload) ? payload : payload.graphs;
        }),
      );
    }
    const graphs = await shardCache.get(path)!;
    const graph = graphs[Number(offset)];
    if (!graph) throw new Error(`Graph offset ${offset} not found in shard: ${path}`);
    return graph;
  }

  const response = await fetch(`/graphs/${path}`);
  if (!response.ok) {
    throw new Error(`Could not load graph file: ${path}`);
  }
  return response.json();
}
