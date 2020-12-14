
# Environment variables

There are several environment variables that can
affect running benchmarks.

## `K2_BENCHMARK_FILTER`

It specifies a regular expression. Benchmark names
that do not match the pattern will be excluded. Only
benchmarks with name matching the pattern will be run.

## `K2_SEED`

It specifies the seed for the random generator. If
it is non-zero, then the results are reproducible.
If it is not set or its value is 0, then every time
the benchmark runs, it uses a different sets of data.
