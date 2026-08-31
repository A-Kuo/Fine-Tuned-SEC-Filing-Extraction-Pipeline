insert into public.model_metrics (
    model_version,
    metric_name,
    metric_value,
    sample_size,
    metadata
) values
('llama-sec-v1', 'field_accuracy', 0.94, 20, '{"split":"synthetic_test"}'::jsonb),
('llama-sec-v1', 'p50_latency_ms', 320, 10, '{"hardware":"A100"}'::jsonb),
('llama-sec-v1', 'throughput_docs_per_min', 60, 10, '{"hardware":"A100"}'::jsonb);