import { describe, expect, it } from "vitest";
import {
  computeSession,
  deriveSession,
  etWallToEpoch,
  nyseEarlyCloses,
  nyseHolidays,
} from "../market-hours";

const utc = (iso: string) => Date.parse(iso);
const dates = (m: Map<string, string>) => [...m.keys()].sort();

describe("nyseHolidays", () => {
  it("2025 matches the published NYSE calendar", () => {
    expect(dates(nyseHolidays(2025))).toEqual([
      "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
      "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    ]);
  });

  it("2026 observes Independence Day on Fri Jul 3 (Jul 4 is a Saturday)", () => {
    expect(dates(nyseHolidays(2026))).toEqual([
      "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
      "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    ]);
  });

  it("2027 applies Sat -> Fri and Sun -> Mon observance", () => {
    const h = nyseHolidays(2027);
    expect(h.has("2027-06-18")).toBe(true); // Juneteenth is Saturday Jun 19
    expect(h.has("2027-07-05")).toBe(true); // Jul 4 is a Sunday
    expect(h.has("2027-12-24")).toBe(true); // Christmas is a Saturday
    expect(h.has("2027-03-26")).toBe(true); // Good Friday (Easter Mar 28)
  });

  it("does not observe a Saturday Jan 1 on the prior Friday", () => {
    expect(nyseHolidays(2027).has("2027-12-31")).toBe(false);
    expect(nyseHolidays(2028).has("2028-01-01")).toBe(false);
  });
});

describe("nyseEarlyCloses", () => {
  it("2026: day after Thanksgiving and Christmas Eve only (Jul 3 is a full holiday)", () => {
    expect(dates(nyseEarlyCloses(2026))).toEqual(["2026-11-27", "2026-12-24"]);
  });

  it("2025 includes Jul 3 because Jul 4 is a Friday", () => {
    expect(dates(nyseEarlyCloses(2025))).toEqual(["2025-07-03", "2025-11-28", "2025-12-24"]);
  });
});

describe("etWallToEpoch", () => {
  it("handles both sides of the 2026 DST changes", () => {
    expect(etWallToEpoch(2026, 3, 6, 570)).toBe(Date.UTC(2026, 2, 6, 14, 30)); // EST
    expect(etWallToEpoch(2026, 3, 9, 570)).toBe(Date.UTC(2026, 2, 9, 13, 30)); // EDT
    expect(etWallToEpoch(2026, 10, 30, 570)).toBe(Date.UTC(2026, 9, 30, 13, 30)); // EDT
    expect(etWallToEpoch(2026, 11, 2, 570)).toBe(Date.UTC(2026, 10, 2, 14, 30)); // EST
  });
});

describe("computeSession", () => {
  const cases: Array<[string, string, string, string?]> = [
    ["Fri 10:45 ET is regular", "2026-09-18T14:45:57Z", "REGULAR", "Closes 4:00 PM ET"],
    ["Fri 08:00 ET is pre-market", "2026-09-18T12:00:00Z", "PRE", "Opens 9:30 AM ET"],
    ["Fri 17:00 ET is after-hours", "2026-09-18T21:00:00Z", "POST", "Opens Mon 9:30 AM ET"],
    ["Fri 21:00 ET is closed", "2026-09-19T01:00:00Z", "CLOSED", "Opens Mon 9:30 AM ET"],
    ["Saturday is closed", "2026-09-19T15:00:00Z", "CLOSED", "Opens Mon 9:30 AM ET"],
    ["Sunday is closed", "2026-09-20T15:00:00Z", "CLOSED", "Opens Mon 9:30 AM ET"],
    ["09:29 ET is still pre-market", "2026-09-18T13:29:00Z", "PRE"],
    ["09:30 ET is regular", "2026-09-18T13:30:00Z", "REGULAR"],
    ["15:59 ET is regular", "2026-09-18T19:59:00Z", "REGULAR"],
    ["16:00 ET is after-hours", "2026-09-18T20:00:00Z", "POST"],
    ["03:59 ET is closed", "2026-09-18T07:59:00Z", "CLOSED"],
  ];
  it.each(cases)("%s", (_name, iso, state, label) => {
    const s = computeSession(utc(iso));
    expect(s.state).toBe(state);
    expect(s.source).toBe("computed");
    if (label) expect(s.nextChangeLabel).toBe(label);
  });

  it.each([
    ["2026-01-01T15:00:00Z", "New Year's Day"],
    ["2026-04-03T15:00:00Z", "Good Friday"],
    ["2026-07-03T15:00:00Z", "Independence Day"],
    ["2026-09-07T15:00:00Z", "Labor Day"],
    ["2026-11-26T15:00:00Z", "Thanksgiving Day"],
    ["2026-12-25T15:00:00Z", "Christmas Day"],
  ])("%s is a full-day holiday closure", (iso, name) => {
    const s = computeSession(utc(iso));
    expect(s.state).toBe("CLOSED");
    expect(s.holiday).toBe(name);
  });

  it("holiday next-open skips to the next trading day", () => {
    const s = computeSession(utc("2026-09-07T15:00:00Z")); // Labor Day Monday
    expect(s.nextChangeLabel).toBe("Opens Tue 9:30 AM ET");
    expect(s.nextChangeMs).toBe(etWallToEpoch(2026, 9, 8, 570));
  });

  it("early close day: regular until 13:00 ET, then after-hours", () => {
    const before = computeSession(utc("2026-11-27T17:59:00Z")); // 12:59 EST
    expect(before.state).toBe("REGULAR");
    expect(before.earlyClose).toBe(true);
    expect(before.nextChangeLabel).toBe("Closes 1:00 PM ET");
    expect(computeSession(utc("2026-11-27T18:30:00Z")).state).toBe("POST"); // 13:30 EST
    expect(computeSession(utc("2026-11-27T22:01:00Z")).state).toBe("CLOSED"); // 17:01 EST
  });

  it("crosses the spring-forward boundary without shifting the open", () => {
    expect(computeSession(utc("2026-03-06T14:29:00Z")).state).toBe("PRE"); // Fri, EST
    expect(computeSession(utc("2026-03-06T14:30:00Z")).state).toBe("REGULAR");
    expect(computeSession(utc("2026-03-09T13:29:00Z")).state).toBe("PRE"); // Mon, EDT
    expect(computeSession(utc("2026-03-09T13:30:00Z")).state).toBe("REGULAR");
  });

  it("crosses the fall-back boundary without shifting the open", () => {
    expect(computeSession(utc("2026-10-30T13:30:00Z")).state).toBe("REGULAR"); // Fri, EDT
    expect(computeSession(utc("2026-11-02T14:29:00Z")).state).toBe("PRE"); // Mon, EST
    expect(computeSession(utc("2026-11-02T14:30:00Z")).state).toBe("REGULAR");
  });
});

// Built from a live capture of
// query1.finance.yahoo.com/v8/finance/chart/%5EGSPC on 2026-09-18 10:45 ET.
const LIVE_META = {
  regularMarketTime: 1789742757, // 2026-09-18T14:45:57Z
  currentTradingPeriod: {
    pre: { start: 1789718400, end: 1789738200 },
    regular: { start: 1789738200, end: 1789761600 },
    post: { start: 1789761600, end: 1789776000 },
  },
};
const sec = (s: number) => s * 1000;

describe("deriveSession (Yahoo chart meta)", () => {
  it("live payload one second after the last trade is REGULAR, sourced from yahoo", () => {
    const s = deriveSession(sec(1789742758), LIVE_META);
    expect(s.state).toBe("REGULAR");
    expect(s.source).toBe("yahoo");
    expect(s.lastTradeMs).toBe(sec(1789742757));
    expect(s.nextChangeLabel).toBe("Closes 4:00 PM ET");
  });

  it("a stale last trade inside the regular window means trading is not happening", () => {
    const s = deriveSession(sec(1789742757 + 30 * 60), LIVE_META);
    expect(s.state).toBe("CLOSED");
    expect(s.source).toBe("yahoo");
    expect(s.nextChangeLabel).toMatch(/^Opens /);
  });

  it("no trade yet today and past the open grace period is CLOSED", () => {
    const meta = { ...LIVE_META, regularMarketTime: LIVE_META.currentTradingPeriod.regular.start - 90000 };
    const s = deriveSession(sec(LIVE_META.currentTradingPeriod.regular.start + 600), meta);
    expect(s.state).toBe("CLOSED");
  });

  it("within 3 minutes of the open, yesterday's trade timestamp is tolerated", () => {
    const meta = { ...LIVE_META, regularMarketTime: LIVE_META.currentTradingPeriod.regular.start - 90000 };
    const s = deriveSession(sec(LIVE_META.currentTradingPeriod.regular.start + 60), meta);
    expect(s.state).toBe("REGULAR");
  });

  it("pre-market and after-hours windows map to PRE and POST", () => {
    expect(deriveSession(sec(1789730000), LIVE_META).state).toBe("PRE");
    expect(deriveSession(sec(1789770000), LIVE_META).state).toBe("POST");
  });

  it("outside every window is CLOSED", () => {
    expect(deriveSession(sec(1789780000), LIVE_META).state).toBe("CLOSED");
  });

  it("a holiday is vetoed by the calendar even though Yahoo still publishes windows", () => {
    const day = (mm: number) => etWallToEpoch(2026, 7, 3, mm) / 1000;
    const meta = {
      regularMarketTime: etWallToEpoch(2026, 7, 2, 16 * 60) / 1000, // Thursday's close
      currentTradingPeriod: {
        pre: { start: day(240), end: day(570) },
        regular: { start: day(570), end: day(960) },
        post: { start: day(960), end: day(1200) },
      },
    };
    const s = deriveSession(etWallToEpoch(2026, 7, 3, 10 * 60), meta);
    expect(s.state).toBe("CLOSED");
    expect(s.holiday).toBe("Independence Day");
  });

  it("early-close afternoon defers to the calendar (after-hours), not Yahoo's 16:00 window", () => {
    const day = (mm: number) => etWallToEpoch(2026, 11, 27, mm) / 1000;
    const meta = {
      regularMarketTime: day(13 * 60),
      currentTradingPeriod: {
        pre: { start: day(240), end: day(570) },
        regular: { start: day(570), end: day(960) },
        post: { start: day(960), end: day(1200) },
      },
    };
    expect(deriveSession(etWallToEpoch(2026, 11, 27, 13 * 60 + 30), meta).state).toBe("POST");
  });

  it.each([
    ["null", null],
    ["undefined", undefined],
    ["empty object", {}],
    ["missing regularMarketTime", { currentTradingPeriod: LIVE_META.currentTradingPeriod }],
    ["non-numeric periods", { regularMarketTime: 1, currentTradingPeriod: { pre: {}, regular: {}, post: {} } }],
  ])("falls back to the computed calendar for %s meta", (_name, meta) => {
    const s = deriveSession(utc("2026-09-18T14:45:57Z"), meta);
    expect(s.source).toBe("computed");
    expect(s.state).toBe("REGULAR");
    expect(s.lastTradeMs).toBeUndefined();
  });
});
