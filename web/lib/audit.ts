import { useSyncExternalStore } from "react";
import type { AuditEvent } from "./types";

// Client-only. Events recorded here live in this browser's localStorage: they
// describe what *this* user did in this UI, they are not a tamper-evident or
// server-side audit log, and the Audit Trails page says so.

export const AUDIT_STORAGE_KEY = "edgarx.audit.v1";
export const AUDIT_MAX_EVENTS = 200;
const LOCAL_ACTOR = "You (this browser)";
const EMPTY: AuditEvent[] = [];

let cache: AuditEvent[] | null = null;
const listeners = new Set<() => void>();

function storage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function isEvent(value: unknown): value is AuditEvent {
  const v = value as Record<string, unknown> | null;
  return (
    !!v &&
    typeof v.id === "string" &&
    typeof v.ts === "string" &&
    typeof v.actor === "string" &&
    typeof v.action === "string" &&
    typeof v.object === "string" &&
    typeof v.detail === "string" &&
    v.source === "local"
  );
}

function load(): AuditEvent[] {
  const s = storage();
  if (!s) return EMPTY;
  try {
    const raw = s.getItem(AUDIT_STORAGE_KEY);
    if (!raw) return EMPTY;
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isEvent) : EMPTY;
  } catch {
    return EMPTY;
  }
}

function emit(): void {
  listeners.forEach((listener) => listener());
}

// useSyncExternalStore requires the same reference until the data changes.
export function getAuditSnapshot(): AuditEvent[] {
  if (cache === null) cache = load();
  return cache;
}

export function getAuditServerSnapshot(): AuditEvent[] {
  return EMPTY;
}

export function subscribeAudit(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === AUDIT_STORAGE_KEY || event.key === null) {
      cache = null;
      emit();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

function newId(): string {
  const c = globalThis.crypto;
  return typeof c?.randomUUID === "function" ? c.randomUUID() : `local-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function recordAuditEvent(input: { action: string; object: string; detail: string }): AuditEvent {
  const event: AuditEvent = {
    id: newId(),
    ts: new Date().toISOString(),
    actor: LOCAL_ACTOR,
    source: "local",
    ...input,
  };
  const next = [event, ...getAuditSnapshot()].slice(0, AUDIT_MAX_EVENTS);
  cache = next;
  try {
    storage()?.setItem(AUDIT_STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Quota exceeded or storage blocked (private mode): keep the event in memory for this session.
  }
  emit();
  return event;
}

export function resetAuditCacheForTests(): void {
  cache = null;
}

export function useLocalAuditEvents(): AuditEvent[] {
  return useSyncExternalStore(subscribeAudit, getAuditSnapshot, getAuditServerSnapshot);
}
