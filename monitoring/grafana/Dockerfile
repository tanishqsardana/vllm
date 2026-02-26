FROM grafana/grafana:11.2.2

USER root
COPY provisioning /etc/grafana/provisioning
COPY dashboards /var/lib/grafana/dashboards
RUN chown -R 472:0 /etc/grafana/provisioning /var/lib/grafana/dashboards

USER 472
