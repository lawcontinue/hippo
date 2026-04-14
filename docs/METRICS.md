# Prometheus Metrics for Hippo 🦛

Hippo v0.2.0+ includes built-in Prometheus metrics support for monitoring inference performance and resource usage.

## Overview

The metrics endpoint exposes the following metrics:

### Inference Metrics

- **`hippo_inference_requests_total`** (Counter)
  - Total number of inference requests
  - Labels: `model`, `endpoint`, `status`
  - Example: `hippo_inference_requests_total{model="llama3.2",endpoint="/api/generate",status="success"} 42`

- **`hippo_inference_duration_seconds`** (Histogram)
  - Inference request duration in seconds
  - Labels: `model`, `endpoint`
  - Buckets: 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0 seconds
  - Example: `hippo_inference_duration_seconds_bucket{model="llama3.2",endpoint="/api/chat",le="1.0"} 35`

- **`hippo_tokens_generated_total`** (Counter)
  - Total number of tokens generated (reserved for future use)
  - Labels: `model`

### Model Metrics

- **`hippo_models_loaded`** (Gauge)
  - Number of models currently loaded
  - Labels: `model`
  - Example: `hippo_models_loaded{model="llama3.2"} 1`

- **`hippo_memory_usage_bytes`** (Gauge)
  - Memory usage in bytes by model
  - Labels: `model`
  - Example: `hippo_memory_usage_bytes{model="llama3.2"} 4215705600`

## Configuration

Metrics are **enabled by default** in Hippo v0.2.0+.

### Config File (`~/.hippo/config.yaml`)

```yaml
metrics_enabled: true  # Enable/disable /metrics endpoint
metrics_path: "/metrics"  # Endpoint path
```

### Environment Variables

Metrics can also be controlled via environment variables (future).

## Security

### Default Behavior

- ✅ **Enabled by default** for localhost monitoring
- ✅ **No authentication required** for `/metrics` endpoint
- ⚠️ **Exposed to all network interfaces** if `server.host = 0.0.0.0`

### Security Best Practices

1. **Bind to localhost only** (default):
   ```yaml
   server:
     host: "127.0.0.1"
   ```

2. **Disable metrics in production**:
   ```yaml
   metrics_enabled: false
   ```

3. **Use reverse proxy authentication** (nginx, traefik):
   ```nginx
   location /metrics {
     auth_basic "Prometheus Metrics";
     auth_basic_user_file /etc/nginx/.htpasswd;
     proxy_pass http://localhost:11434/metrics;
   }
   ```

4. **Network firewall**:
   - Block port 11434 from external access
   - Use VPN for internal monitoring

## Usage

### Access Metrics

```bash
# curl
curl http://localhost:11434/metrics

# wget
wget -qO- http://localhost:11434/metrics

# Prometheus scrape config
scrape_configs:
  - job_name: 'hippo'
    static_configs:
      - targets: ['localhost:11434']
    metrics_path: '/metrics'
```

### Example Metrics Output

```
# HELP hippo_inference_requests_total Total number of inference requests
# TYPE hippo_inference_requests_total counter
hippo_inference_requests_total{endpoint="/api/generate",model="llama3.2",status="success"} 42
hippo_inference_requests_total{endpoint="/api/generate",model="llama3.2",status="error"} 2
hippo_inference_requests_total{endpoint="/api/chat",model="llama3.2",status="success"} 15

# HELP hippo_inference_duration_seconds Inference request duration in seconds
# TYPE hippo_inference_duration_seconds histogram
hippo_inference_duration_seconds_bucket{endpoint="/api/generate",model="llama3.2",le="0.1"} 0
hippo_inference_duration_seconds_bucket{endpoint="/api/generate",model="llama3.2",le="0.5"} 5
hippo_inference_duration_seconds_bucket{endpoint="/api/generate",model="llama3.2",le="1.0"} 35
hippo_inference_duration_seconds_bucket{endpoint="/api/generate",model="llama3.2",le="5.0"} 42
hippo_inference_duration_seconds_sum{endpoint="/api/generate",model="llama3.2"} 87.5
hippo_inference_duration_seconds_count{endpoint="/api/generate",model="llama3.2"} 42

# HELP hippo_models_loaded Number of models currently loaded
# TYPE hippo_models_loaded gauge
hippo_models_loaded{model="llama3.2"} 1

# HELP hippo_memory_usage_bytes Memory usage in bytes by model
# TYPE hippo_memory_usage_bytes gauge
hippo_memory_usage_bytes{model="llama3.2"} 4215705600
```

## Prometheus Grafana Dashboard

### Example Queries

**Request rate:**
```promql
rate(hippo_inference_requests_total{status="success"}[5m])
```

**P95 latency:**
```promql
histogram_quantile(0.95,
  rate(hippo_inference_duration_seconds_bucket[5m])
)
```

**Memory usage:**
```promql
hippo_memory_usage_bytes / 1024^3  # in GB
```

**Error rate:**
```promql
rate(hippo_inference_requests_total{status="error"}[5m]) /
rate(hippo_inference_requests_total[5m])
```

## Testing

### Manual Test Script

```bash
# Run the test script
python test_metrics_manual.py
```

### Unit Tests

```bash
# Run metrics tests
pytest tests/test_metrics.py -v

# Run all tests
pytest tests/ -v
```

## Troubleshooting

### Metrics endpoint returns 403

**Problem:** `curl http://localhost:11434/metrics` returns `403 Forbidden`

**Solution:** Enable metrics in config:
```yaml
metrics_enabled: true
```

### Metrics not updating

**Problem:** Metrics counters stay at zero

**Possible causes:**
1. No inference requests made yet
2. Model not loaded
3. Metrics disabled

**Solution:** Make a test request:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "hi"
}'
```

### High memory usage metrics

**Problem:** `hippo_memory_usage_bytes` shows very high values

**Solution:** This is the model file size, not RAM usage. To reduce:
```bash
hippo unload <model-name>
```

## Implementation Details

### Files Changed

- `hippo/metrics.py` — Metric definitions
- `hippo/routers/metrics.py` — `/metrics` endpoint
- `hippo/config.py` — Metrics configuration
- `hippo/api.py` — Router registration
- `hippo/routers/inference.py` — Metrics integration
- `hippo/model_manager.py` — Model lifecycle metrics

### Dependencies

- `prometheus-client>=0.20.0` — Pure Python Prometheus client

### Performance Impact

- **Minimal overhead** (~0.1ms per request)
- **No external services** required
- **Thread-safe** metrics collection

## Future Enhancements

- [ ] Token count tracking (`hippo_tokens_generated_total`)
- [ ] GPU memory metrics (if applicable)
- [ ] Request queue depth
- [ ] Cache hit/miss rates
- [ ] Custom labels support

## License

MIT License — see LICENSE file for details.
