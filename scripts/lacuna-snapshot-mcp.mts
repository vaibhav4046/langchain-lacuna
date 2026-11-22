/**
 * Lacuna's own MCP server, reading the graph snapshot checked into the Lacuna repository
 * instead of a live HydraDB node. Used by this package's integration tests and offline example.
 *
 *   LACUNA_DIR=/path/to/lacuna npx --prefix "$LACUNA_DIR" tsx scripts/lacuna-snapshot-mcp.mts
 *
 * Every module comes from the Lacuna checkout; this file only wires them together the way
 * Lacuna's scripts/mcp.ts (stdio server) and src/snapshot/serve.ts (snapshot replay) do.
 */
import { pathToFileURL } from 'node:url';
import { join, resolve } from 'node:path';

const dir = process.env['LACUNA_DIR'];
if (!dir) {
  process.stderr.write('Set LACUNA_DIR to a Lacuna checkout with `npm ci` done.\n');
  process.exit(2);
}
const root = resolve(dir);
const load = (path: string) => import(pathToFileURL(join(root, path)).href);

// Lacuna's stdio server keeps stdout for JSON-RPC frames only.
const toStderr = (...parts: unknown[]): void => {
  process.stderr.write(parts.map(String).join(' ') + '\n');
};
console.log = console.info = console.debug = console.warn = toStderr;

const { StdioServerTransport } = await import(
  pathToFileURL(join(root, 'node_modules/@modelcontextprotocol/sdk/dist/esm/server/stdio.js')).href
);
const { HydraClient } = await load('src/hydra/client.ts');
const { NodeSource } = await load('src/hydra/node-source.ts');
const { loadSnapshot, snapshotTransport } = await load('src/snapshot/replay.ts');
const { describeNode } = await load('src/view/proof.ts');
const { createMcpServer } = await load('src/mcp/server.ts');

const snapshot = loadSnapshot(root);
const config = {
  baseUrl: 'http://127.0.0.1:1', // never opened: the snapshot transport answers every read
  namespace: snapshot.node.namespace,
  graph: snapshot.node.graph,
  cell: snapshot.node.cell,
  token: 'snapshot-replay-unused',
};
const client = new HydraClient(config, { fetch: snapshotTransport(snapshot) });
const server = createMcpServer({ source: new NodeSource(client), node: describeNode(config), store: 'node' });
await server.connect(new StdioServerTransport());
process.stderr.write(`lacuna snapshot MCP on stdio (exported ${snapshot.exportedAt})\n`);
