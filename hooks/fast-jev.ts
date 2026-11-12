import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  retryGapPercent: 10,
  minReductionRatio: 0.25,
  builtinFallback: false,
  model: DEFAULT_MODEL,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  compactAtPercent: number;
  retryGapPercent: number;
  minReductionRatio: number;
  /** When true, the main conversation gets the built-in summary instead of a skip. */
  builtinFallback: boolean;
  model: string;
  /** Where results are logged; unset or empty is `~/.claude/fast-jev-compaction-log.jsonl`, `off` is none. */
  logFile?: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionBoolean(options: PluginOptions, key: string, fallback: boolean): boolean {
  const value = options[key];
  return typeof value === 'boolean' ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** A comma-separated list of names; empty entries are ignored, so `''` is none. */
function optionList(options: PluginOptions, key: string): string[] | undefined {
  const value = options[key];
  if (typeof value !== 'string') return undefined;
  return value
    .split(',')
    .map((name) => name.trim())
    .filter((name) => name.length > 0);
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    retryGapPercent: optionNumber(options, 'retryGapPercent', HOOK_DEFAULTS.retryGapPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    builtinFallback: optionBoolean(options, 'builtinFallback', HOOK_DEFAULTS.builtinFallback),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const logFile = optionString(options, 'logFile')?.trim();
  if (logFile) config.logFile = logFile;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  const pinnedTools = optionList(options, 'pinnedTools');
  if (pinnedTools) config.pinnedTools = pinnedTools;
  return config;
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(fetchFn: HookFetch, apiKey: string, model: string): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** Runs the library over a session transcript; throws when the key is missing or Jev fails. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
): Promise<SessionCompaction> {
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  const result = await compact(messages, jevAsker(fetchFn, config.apiKey, config.model), config);
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

/** How many compaction results the log file keeps; older lines are dropped. */
export const LOG_MAX_ENTRIES = 10;
const LOG_NAME = '.claude/fast-jev-compaction-log.jsonl';

export type Outcome = 'kept' | 'skipped' | 'fallback';

export type LogEntry = {
  time: string;
  session?: string;
  cwd?: string;
  trigger?: string;
  agentId?: string;
  outcome: Outcome;
  text: string;
};

/** Appends `entry` as one JSON line to `existing` and keeps the last `max` lines. */
export function appendLogEntry(existing: string, entry: LogEntry, max: number = LOG_MAX_ENTRIES): string {
  const lines = existing.split('\n').filter((line) => line.trim().length > 0);
  lines.push(JSON.stringify(entry));
  return `${lines.slice(-max).join('\n')}\n`;
}

type HookContext = {
  ui: {
    log: (text: string) => void;
    toast: (text: string, options?: { timeoutMs?: number }) => void;
  };
  env: { get: (name: string) => Promise<string | undefined> };
  clock: { now: () => Promise<number> };
  session: { id: () => Promise<string>; cwd: () => Promise<string> };
  fs: { read: (path: string) => Promise<string>; write: (path: string, text: string) => Promise<void> };
};

/** Awaits `get`, or undefined when it throws or rejects. */
async function optional<T>(get: () => Promise<T>): Promise<T | undefined> {
  try {
    return await get();
  } catch {
    return undefined;
  }
}

/** The log file's path: `logFile` when set, else `~/.claude/...`; null when it is `off`. */
async function logPath($: HookContext, config: Pick<HookConfig, 'logFile'>): Promise<string | null> {
  if (config.logFile) return config.logFile.toLowerCase() === 'off' ? null : config.logFile;
  const home = (await optional(() => $.env.get('USERPROFILE'))) ?? (await optional(() => $.env.get('HOME')));
  return home ? `${home}/${LOG_NAME}` : null;
}

/**
 * Shows `text` and records it in the log file, so a client that hides
 * toasts (T3 Code) can still tell what the last compactions did. A failure to
 * write the log is noted in the UI log and never stops the compaction.
 */
async function report(
  $: HookContext,
  config: Pick<HookConfig, 'logFile'>,
  event: { trigger?: string; agentId?: string },
  outcome: Outcome,
  text: string,
): Promise<void> {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
  try {
    const path = await logPath($, config);
    if (!path) return;
    const now = (await optional(() => $.clock.now())) ?? Date.now();
    const entry: LogEntry = { time: new Date(now).toISOString(), outcome, text };
    const session = await optional(() => $.session.id());
    if (session) entry.session = session;
    const cwd = await optional(() => $.session.cwd());
    if (cwd) entry.cwd = cwd;
    if (event.trigger) entry.trigger = event.trigger;
    if (event.agentId) entry.agentId = event.agentId;
    const existing = (await optional(() => $.fs.read(path))) ?? '';
    await $.fs.write(path, appendLogEntry(existing, entry));
  } catch (error) {
    $.ui.log(`compaction log not written (${error instanceof Error ? error.message : String(error)})`);
  }
}

/**
 * When Jev cannot compact, or would save less than minReductionRatio, the
 * main conversation is left as it is (`{ skip }`) rather than summarized: a
 * summary loses exact text. `builtinFallback` restores the built-in summary
 * for the main conversation. A helper's own transcript (`agentId` set) always
 * gets the built-in summary, because a helper cannot be steered and would
 * otherwise run out of context.
 */
async function giveUp(
  $: HookContext,
  config: Pick<HookConfig, 'builtinFallback' | 'logFile'>,
  event: { trigger?: string; agentId?: string },
  next: (e: any) => Promise<any>,
  reason: string,
): Promise<any> {
  if (event.agentId !== undefined) {
    await report($, config, event, 'fallback', `fallback to built-in summary for helper ${event.agentId} (${reason})`);
    return next(event);
  }
  if (config.builtinFallback) {
    await report($, config, event, 'fallback', `fallback to built-in summary (${reason})`);
    return next(event);
  }
  const text = `compaction skipped, conversation unchanged (${reason})`;
  await report($, config, event, 'skipped', text);
  return { skip: text };
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;
  // After a compaction that left the context at or above compactAtPercent,
  // the next automatic try waits until it reaches this percent, so a session
  // with little to trim is not compacted again after every turn.
  let retryAtPercent = 0;

  on('session.compact', async ($, event, next) => {
    try {
      const config = { ...configured, apiKey: await getApiKey($, configured) };
      const { result, messages } = await compactSession(event.messages, config, async (url, init) => {
        const response = await $.http.fetch(url, init);
        return { status: response.status, ok: response.ok, text: response.text };
      });
      for (const line of decisionLogLines(result)) $.ui.log(line);
      // A trim this small frees too little to be worth replacing the history.
      if (reductionRatio(result) < config.minReductionRatio) {
        return giveUp(
          $,
          config,
          event,
          next,
          `below ${percent(config.minReductionRatio)} minimum: ${summarize(result)}`,
        );
      }
      await report(
        $,
        config,
        event,
        'kept',
        `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`,
      );
      return { messages };
    } catch (error) {
      return giveUp($, configured, event, next, error instanceof Error ? error.message : String(error));
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting) return next(event);
    try {
      const before = (await $.session.usage()).context.percent ?? 0;
      if (before < configured.compactAtPercent) {
        retryAtPercent = 0;
        return next(event);
      }
      if (before < retryAtPercent) return next(event);
      compacting = true;
      await $.session.compact();
      const after = (await $.session.usage()).context.percent ?? 0;
      retryAtPercent =
        after >= configured.compactAtPercent ? after + configured.retryGapPercent : 0;
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
