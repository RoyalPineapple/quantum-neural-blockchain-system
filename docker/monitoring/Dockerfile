# Use Prometheus base image
FROM prom/prometheus:latest

# Copy Prometheus configuration
COPY config/prometheus/prometheus.yml /etc/prometheus/prometheus.yml

# Set ownership for config files
USER root
RUN chown -R nobody:nobody /etc/prometheus

# Switch back to non-root user
USER nobody

# Expose Prometheus port
EXPOSE 9090

# Start Prometheus with configuration
ENTRYPOINT [ "/bin/prometheus" ]
CMD        [ "--config.file=/etc/prometheus/prometheus.yml", \
            "--storage.tsdb.path=/prometheus", \
            "--web.console.libraries=/usr/share/prometheus/console_libraries", \
            "--web.console.templates=/usr/share/prometheus/consoles" ]
