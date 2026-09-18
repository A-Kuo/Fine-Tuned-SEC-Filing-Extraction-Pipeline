import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AUDIT_MAX_EVENTS,
  AUDIT_STORAGE_KEY,
  getAuditServerSnapshot,
  getAuditSnapshot,
  recordAuditEvent,
  resetAuditCacheForTests,
  subscribeAudit,
} from "../audit";

function fakeStorage(seed: Record<string, string> = {}): Storage {
  const data = new Map(Object.entries(seed));
  return {
    get length() { return data.size; },
    clear: () => data.clear(),
    getItem: (k) => data.get(k) ?? null,
    key: (i) => [...data.keys()][i] ?? null,
    removeItem: (k) => void data.delete(k),
    setItem: (k, v) => void data.set(k, v),
  };
}

beforeEach(() => {
  vi.stubGlobal("localStorage", fakeStorage());
  resetAuditCacheForTests();
});
afterEach(() => vi.unstubAllGlobals());

const sample = { action: "export.csv", object: "Regulatory Filings", detail: "12 rows" };

describe("recordAuditEvent", () => {
  it("records a local, timestamped event with the caller's fields", () => {
    const e = recordAuditEvent(sample);
    expect(e).toMatchObject({ ...sample, source: "local", actor: "You (this browser)" });
    expect(e.id).toBeTruthy();
    expect(Number.isNaN(Date.parse(e.ts))).toBe(false);
  });

  it("keeps newest first and persists to localStorage", () => {
    recordAuditEvent({ ...sample, detail: "first" });
    recordAuditEvent({ ...sample, detail: "second" });
    expect(getAuditSnapshot().map((e) => e.detail)).toEqual(["second", "first"]);
    const stored = JSON.parse(localStorage.getItem(AUDIT_STORAGE_KEY)!);
    expect(stored.map((e: { detail: string }) => e.detail)).toEqual(["second", "first"]);
  });

  it("caps the log, dropping the oldest events", () => {
    for (let i = 0; i < AUDIT_MAX_EVENTS + 5; i++) recordAuditEvent({ ...sample, detail: `n${i}` });
    const events = getAuditSnapshot();
    expect(events).toHaveLength(AUDIT_MAX_EVENTS);
    expect(events[0].detail).toBe(`n${AUDIT_MAX_EVENTS + 4}`);
    expect(events.at(-1)!.detail).toBe("n5");
  });

  it("still holds the event in memory when storage rejects writes", () => {
    vi.stubGlobal("localStorage", { ...fakeStorage(), setItem: () => { throw new Error("QuotaExceededError"); } });
    resetAuditCacheForTests();
    expect(() => recordAuditEvent(sample)).not.toThrow();
    expect(getAuditSnapshot()).toHaveLength(1);
  });
});

describe("getAuditSnapshot", () => {
  it("returns the same reference until something changes (required by useSyncExternalStore)", () => {
    const a = getAuditSnapshot();
    expect(getAuditSnapshot()).toBe(a);
    recordAuditEvent(sample);
    const b = getAuditSnapshot();
    expect(b).not.toBe(a);
    expect(getAuditSnapshot()).toBe(b);
  });

  it("loads previously stored events", () => {
    const stored = [{ id: "x", ts: "2026-09-18T10:00:00.000Z", actor: "You (this browser)", action: "print.ledger", object: "o", detail: "d", source: "local" }];
    vi.stubGlobal("localStorage", fakeStorage({ [AUDIT_STORAGE_KEY]: JSON.stringify(stored) }));
    resetAuditCacheForTests();
    expect(getAuditSnapshot()).toEqual(stored);
  });

  it("survives corrupt JSON, a non-array value, and malformed entries", () => {
    for (const raw of ["{not json", '{"a":1}', "null"]) {
      vi.stubGlobal("localStorage", fakeStorage({ [AUDIT_STORAGE_KEY]: raw }));
      resetAuditCacheForTests();
      expect(getAuditSnapshot()).toEqual([]);
    }
    const good = { id: "g", ts: "t", actor: "a", action: "x", object: "o", detail: "d", source: "local" };
    vi.stubGlobal("localStorage", fakeStorage({ [AUDIT_STORAGE_KEY]: JSON.stringify([good, { id: 1 }, "str", null]) }));
    resetAuditCacheForTests();
    expect(getAuditSnapshot()).toEqual([good]);
  });

  it("does not trust stored events claiming to be system events", () => {
    const forged = { id: "f", ts: "t", actor: "svc-review", action: "x", object: "o", detail: "d", source: "system" };
    vi.stubGlobal("localStorage", fakeStorage({ [AUDIT_STORAGE_KEY]: JSON.stringify([forged]) }));
    resetAuditCacheForTests();
    expect(getAuditSnapshot()).toEqual([]);
  });

  it("is empty when localStorage is unavailable", () => {
    vi.stubGlobal("localStorage", undefined);
    resetAuditCacheForTests();
    expect(getAuditSnapshot()).toEqual([]);
  });

  it("the server snapshot is always empty and referentially stable", () => {
    recordAuditEvent(sample);
    expect(getAuditServerSnapshot()).toEqual([]);
    expect(getAuditServerSnapshot()).toBe(getAuditServerSnapshot());
  });
});

describe("subscribeAudit", () => {
  function stubWindow() {
    const handlers = new Set<(e: StorageEvent) => void>();
    vi.stubGlobal("window", {
      addEventListener: (_: string, h: (e: StorageEvent) => void) => handlers.add(h),
      removeEventListener: (_: string, h: (e: StorageEvent) => void) => handlers.delete(h),
    });
    return handlers;
  }

  it("notifies on record and stops after unsubscribe", () => {
    stubWindow();
    const listener = vi.fn();
    const unsubscribe = subscribeAudit(listener);
    recordAuditEvent(sample);
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
    recordAuditEvent(sample);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("reloads from storage and notifies when another tab writes the key", () => {
    const handlers = stubWindow();
    const listener = vi.fn();
    subscribeAudit(listener);
    recordAuditEvent(sample);
    listener.mockClear();

    const other = { id: "o", ts: "t", actor: "a", action: "x", object: "o", detail: "from other tab", source: "local" };
    localStorage.setItem(AUDIT_STORAGE_KEY, JSON.stringify([other]));
    handlers.forEach((h) => h({ key: AUDIT_STORAGE_KEY } as StorageEvent));

    expect(listener).toHaveBeenCalledTimes(1);
    expect(getAuditSnapshot()).toEqual([other]);
  });

  it("ignores storage events for unrelated keys", () => {
    const handlers = stubWindow();
    const listener = vi.fn();
    subscribeAudit(listener);
    handlers.forEach((h) => h({ key: "something-else" } as StorageEvent));
    expect(listener).not.toHaveBeenCalled();
  });
});
