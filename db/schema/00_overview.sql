-- Extensions required for Supabase/Postgres-native operation

create extension if not exists pgcrypto with schema extensions;
create extension if not exists pg_trgm with schema extensions;

-- Optional: enable later when embeddings/search are needed
create extension if not exists vector with schema extensions;