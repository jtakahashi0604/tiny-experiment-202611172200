// Command-line options shared by the scripts: `--name value` and bare `--flag`.

const args = process.argv.slice(2);

// the value after `name` (e.g. "--weights"), or `fallback` when it is absent
export function option({ name, fallback }: { name: string; fallback: string }): string {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
}

export function flag({ name }: { name: string }): boolean {
  return args.includes(name);
}
