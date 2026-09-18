// NYSE session logic. Pure functions only (no network, no globals) so every
// branch is unit-testable with fixed instants. The route handler in
// app/api/market-status/route.ts feeds deriveSession() a Yahoo chart payload.

export type MarketState = "REGULAR" | "PRE" | "POST" | "CLOSED";

export interface MarketSession {
  state: MarketState;
  holiday?: string;
  earlyClose?: boolean;
  nextChangeMs: number | null;
  nextChangeLabel: string;
  source: "yahoo" | "computed";
  lastTradeMs?: number;
  asOfMs: number;
}

export interface YahooChartMeta {
  regularMarketTime: number;
  currentTradingPeriod: {
    pre: { start: number; end: number };
    regular: { start: number; end: number };
    post: { start: number; end: number };
  };
}

interface EtParts {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  weekday: number;
}

const ET_ZONE = "America/New_York";
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

const PRE_OPEN_MIN = 4 * 60;
const REGULAR_OPEN_MIN = 9 * 60 + 30;
const REGULAR_CLOSE_MIN = 16 * 60;
const EARLY_CLOSE_MIN = 13 * 60;
const POST_CLOSE_MIN = 20 * 60;
const EARLY_POST_CLOSE_MIN = 17 * 60;

// Last regular-session trade must be this recent for the market to count as
// actually trading; guards surprise closures and early-close days.
const FRESH_TRADE_SECONDS = 20 * 60;
const OPEN_GRACE_SECONDS = 3 * 60;

const etFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: ET_ZONE,
  hourCycle: "h23",
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  minute: "numeric",
  second: "numeric",
  weekday: "short",
});

export function etParts(epochMs: number): EtParts {
  const parts: Record<string, string> = {};
  for (const p of etFormatter.formatToParts(new Date(epochMs))) parts[p.type] = p.value;
  return {
    year: Number(parts.year),
    month: Number(parts.month),
    day: Number(parts.day),
    hour: Number(parts.hour) % 24,
    minute: Number(parts.minute),
    second: Number(parts.second),
    weekday: WEEKDAYS.indexOf(parts.weekday),
  };
}

function etOffsetMinutes(epochMs: number): number {
  const p = etParts(epochMs);
  const asUtc = Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second);
  return Math.round((asUtc - Math.floor(epochMs / 1000) * 1000) / 60000);
}

export function etWallToEpoch(year: number, month: number, day: number, minuteOfDay: number): number {
  const wall = Date.UTC(year, month - 1, day, 0, minuteOfDay);
  const first = wall - etOffsetMinutes(wall) * 60000;
  return wall - etOffsetMinutes(first) * 60000;
}

export function isoDate(year: number, month: number, day: number): string {
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function weekdayOf(year: number, month: number, day: number): number {
  return new Date(Date.UTC(year, month - 1, day)).getUTCDay();
}

function addDays(year: number, month: number, day: number, delta: number) {
  const d = new Date(Date.UTC(year, month - 1, day + delta));
  return { year: d.getUTCFullYear(), month: d.getUTCMonth() + 1, day: d.getUTCDate() };
}

function nthWeekday(year: number, month: number, weekday: number, n: number): number {
  const first = weekdayOf(year, month, 1);
  return 1 + ((weekday - first + 7) % 7) + (n - 1) * 7;
}

function lastWeekday(year: number, month: number, weekday: number): number {
  const lastDay = new Date(Date.UTC(year, month, 0)).getUTCDate();
  const lastWd = weekdayOf(year, month, lastDay);
  return lastDay - ((lastWd - weekday + 7) % 7);
}

// Anonymous Gregorian algorithm.
function easterSunday(year: number): { month: number; day: number } {
  const a = year % 19;
  const b = Math.floor(year / 100);
  const c = year % 100;
  const d = Math.floor(b / 4);
  const e = b % 4;
  const f = Math.floor((b + 8) / 25);
  const g = Math.floor((b - f + 1) / 3);
  const h = (19 * a + b - d - g + 15) % 30;
  const i = Math.floor(c / 4);
  const k = c % 4;
  const l = (32 + 2 * e + 2 * i - h - k) % 7;
  const m = Math.floor((a + 11 * h + 22 * l) / 451);
  const month = Math.floor((h + l - 7 * m + 114) / 31);
  const day = ((h + l - 7 * m + 114) % 31) + 1;
  return { month, day };
}

// Saturday -> preceding Friday, Sunday -> following Monday.
function observed(year: number, month: number, day: number) {
  const wd = weekdayOf(year, month, day);
  if (wd === 6) return addDays(year, month, day, -1);
  if (wd === 0) return addDays(year, month, day, 1);
  return { year, month, day };
}

const holidayCache = new Map<number, Map<string, string>>();

// Scheduled full-day closures only. Unscheduled closures (e.g. national days
// of mourning) cannot be predicted; the live-trade freshness check in
// deriveSession() is what catches those.
export function nyseHolidays(year: number): Map<string, string> {
  const cached = holidayCache.get(year);
  if (cached) return cached;

  const out = new Map<string, string>();
  const add = (date: { year: number; month: number; day: number }, name: string) => {
    if (date.year === year) out.set(isoDate(date.year, date.month, date.day), name);
  };

  // Jan 1 on a Saturday is not observed on the prior Friday.
  if (weekdayOf(year, 1, 1) !== 6) add(observed(year, 1, 1), "New Year's Day");
  add({ year, month: 1, day: nthWeekday(year, 1, 1, 3) }, "Martin Luther King Jr. Day");
  add({ year, month: 2, day: nthWeekday(year, 2, 1, 3) }, "Washington's Birthday");
  const easter = easterSunday(year);
  add(addDays(year, easter.month, easter.day, -2), "Good Friday");
  add({ year, month: 5, day: lastWeekday(year, 5, 1) }, "Memorial Day");
  if (year >= 2022) add(observed(year, 6, 19), "Juneteenth");
  add(observed(year, 7, 4), "Independence Day");
  add({ year, month: 9, day: nthWeekday(year, 9, 1, 1) }, "Labor Day");
  add({ year, month: 11, day: nthWeekday(year, 11, 4, 4) }, "Thanksgiving Day");
  add(observed(year, 12, 25), "Christmas Day");

  holidayCache.set(year, out);
  return out;
}

export function nyseEarlyCloses(year: number): Map<string, string> {
  const holidays = nyseHolidays(year);
  const out = new Map<string, string>();

  const thanksgiving = nthWeekday(year, 11, 4, 4);
  const dayAfter = addDays(year, 11, thanksgiving, 1);
  out.set(isoDate(dayAfter.year, dayAfter.month, dayAfter.day), "Day after Thanksgiving");

  const dec24 = isoDate(year, 12, 24);
  const dec24Wd = weekdayOf(year, 12, 24);
  if (dec24Wd >= 1 && dec24Wd <= 5 && !holidays.has(dec24)) out.set(dec24, "Christmas Eve");

  // Jul 3 closes early only when Jul 4 falls Tue-Fri (Jul 3 is Mon-Thu).
  const jul3Wd = weekdayOf(year, 7, 3);
  const jul3 = isoDate(year, 7, 3);
  if (jul3Wd >= 1 && jul3Wd <= 4 && !holidays.has(jul3)) out.set(jul3, "Day before Independence Day");

  return out;
}

function tradingDay(year: number, month: number, day: number): boolean {
  const wd = weekdayOf(year, month, day);
  if (wd === 0 || wd === 6) return false;
  return !nyseHolidays(year).has(isoDate(year, month, day));
}

function closeMinuteFor(year: number, month: number, day: number): number {
  return nyseEarlyCloses(year).has(isoDate(year, month, day)) ? EARLY_CLOSE_MIN : REGULAR_CLOSE_MIN;
}

function nextRegularOpenMs(nowMs: number): number {
  const now = etParts(nowMs);
  for (let i = 0; i < 12; i++) {
    const d = addDays(now.year, now.month, now.day, i);
    if (!tradingDay(d.year, d.month, d.day)) continue;
    const openMs = etWallToEpoch(d.year, d.month, d.day, REGULAR_OPEN_MIN);
    if (openMs > nowMs) return openMs;
  }
  return nowMs;
}

function formatEtTime(epochMs: number): string {
  const p = etParts(epochMs);
  const h12 = p.hour % 12 === 0 ? 12 : p.hour % 12;
  const suffix = p.hour < 12 ? "AM" : "PM";
  return `${h12}:${String(p.minute).padStart(2, "0")} ${suffix} ET`;
}

function dayPrefix(targetMs: number, nowMs: number): string {
  const t = etParts(targetMs);
  const n = etParts(nowMs);
  if (t.year === n.year && t.month === n.month && t.day === n.day) return "";
  return `${WEEKDAYS[t.weekday]} `;
}

export function computeSession(nowMs: number): MarketSession {
  const p = etParts(nowMs);
  const date = isoDate(p.year, p.month, p.day);
  const holiday = nyseHolidays(p.year).get(date);
  const isWeekday = p.weekday >= 1 && p.weekday <= 5;
  const earlyClose = nyseEarlyCloses(p.year).has(date);
  const minute = p.hour * 60 + p.minute;

  let state: MarketState = "CLOSED";
  if (isWeekday && !holiday) {
    const closeMin = earlyClose ? EARLY_CLOSE_MIN : REGULAR_CLOSE_MIN;
    const postEnd = earlyClose ? EARLY_POST_CLOSE_MIN : POST_CLOSE_MIN;
    if (minute >= REGULAR_OPEN_MIN && minute < closeMin) state = "REGULAR";
    else if (minute >= PRE_OPEN_MIN && minute < REGULAR_OPEN_MIN) state = "PRE";
    else if (minute >= closeMin && minute < postEnd) state = "POST";
  }

  let nextChangeMs: number | null;
  let nextChangeLabel: string;
  if (state === "REGULAR") {
    nextChangeMs = etWallToEpoch(p.year, p.month, p.day, closeMinuteFor(p.year, p.month, p.day));
    nextChangeLabel = `Closes ${formatEtTime(nextChangeMs)}`;
  } else {
    nextChangeMs = nextRegularOpenMs(nowMs);
    nextChangeLabel = `Opens ${dayPrefix(nextChangeMs, nowMs)}${formatEtTime(nextChangeMs)}`;
  }

  return {
    state,
    holiday,
    earlyClose: earlyClose || undefined,
    nextChangeMs,
    nextChangeLabel,
    source: "computed",
    asOfMs: nowMs,
  };
}

function validMeta(meta: unknown): meta is YahooChartMeta {
  const m = meta as YahooChartMeta | null | undefined;
  const period = m?.currentTradingPeriod;
  const num = (v: unknown) => typeof v === "number" && Number.isFinite(v);
  return (
    !!m &&
    num(m.regularMarketTime) &&
    !!period &&
    num(period.pre?.start) &&
    num(period.pre?.end) &&
    num(period.regular?.start) &&
    num(period.regular?.end) &&
    num(period.post?.start) &&
    num(period.post?.end)
  );
}

// Yahoo's chart meta has no marketState field, so the state is derived: the
// published pre/regular/post windows say when a session is scheduled, the
// NYSE calendar vetoes holidays/weekends/early closes (Yahoo still publishes a
// window on those days), and a fresh regularMarketTime is the evidence that
// trades are actually happening during the regular session.
export function deriveSession(nowMs: number, meta: unknown): MarketSession {
  const calendar = computeSession(nowMs);
  if (!validMeta(meta)) return calendar;

  const nowSec = Math.floor(nowMs / 1000);
  const { pre, regular, post } = meta.currentTradingPeriod;
  const inWindow = (w: { start: number; end: number }) => nowSec >= w.start && nowSec < w.end;
  const lastTradeMs = meta.regularMarketTime * 1000;

  let state: MarketState;
  if (inWindow(regular)) {
    const tradedToday = meta.regularMarketTime >= regular.start;
    const fresh = nowSec - meta.regularMarketTime <= FRESH_TRADE_SECONDS;
    const justOpened = nowSec - regular.start <= OPEN_GRACE_SECONDS;
    if (calendar.state === "REGULAR") {
      state = (tradedToday && fresh) || justOpened ? "REGULAR" : "CLOSED";
    } else {
      state = calendar.state;
    }
  } else if (calendar.state === "CLOSED") {
    state = "CLOSED";
  } else if (inWindow(pre)) {
    state = "PRE";
  } else if (inWindow(post)) {
    state = "POST";
  } else {
    state = "CLOSED";
  }

  const session: MarketSession = { ...calendar, state, source: "yahoo", lastTradeMs };
  if (state === "CLOSED" && calendar.state === "REGULAR") {
    session.nextChangeMs = nextRegularOpenMs(nowMs);
    session.nextChangeLabel = `Opens ${dayPrefix(session.nextChangeMs, nowMs)}${formatEtTime(session.nextChangeMs)}`;
  }
  return session;
}
