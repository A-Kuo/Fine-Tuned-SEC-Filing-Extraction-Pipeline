insert into public.extractions (
    filing_id,
    company_name,
    ticker,
    filing_type,
    filing_date,
    fiscal_year_end,
    revenue,
    net_income,
    total_assets,
    total_liabilities,
    eps,
    sector,
    confidence_score,
    extraction_time_ms,
    model_version,
    method,
    parsed_output
) values (
    '000600181-23-69417296',
    'Berkshire Hathaway Inc.',
    'BRK',
    '10-Q',
    '2023-11-07',
    '2023-09-08',
    362300000000.00,
    61600000000.00,
    1100000000000.00,
    583000000000.00,
    13.19,
    'Financials',
    0.94,
    320,
    'llama-sec-v1',
    'llm',
    '{
      "company_name": "Berkshire Hathaway Inc.",
      "ticker": "BRK",
      "filing_type": "10-Q"
    }'::jsonb
)
on conflict (filing_id) do nothing;