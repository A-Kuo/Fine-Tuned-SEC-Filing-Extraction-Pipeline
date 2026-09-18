import { DATA_AS_OF, getSectors } from "./data";
import { assetOptions, entityOptions, parseFilters, slugify, windowOptions, type FilterState } from "./filters";

export type SearchParams = Record<string, string | string[] | undefined>;

export function filtersFrom(searchParams: SearchParams): FilterState {
  return parseFilters(searchParams, getSectors().map(slugify));
}

export function filterOptions() {
  return {
    asset: assetOptions(),
    entity: entityOptions(getSectors()),
    window: windowOptions(DATA_AS_OF),
  };
}
