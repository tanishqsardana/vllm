#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-appuser}"
APP_HOME="/home/${APP_USER}"
ENABLE_SSHD="${ENABLE_SSHD:-0}"
SSH_PORT="${SSH_PORT:-22}"

if [[ -n "${PUBLIC_KEY:-}" ]]; then
  ENABLE_SSHD="1"
fi

if [[ "$(id -u)" -ne 0 ]]; then
  exec /app/entrypoint.sh
fi

mkdir -p /cache "${APP_HOME}" /app
chown -R "${APP_USER}:${APP_USER}" /cache "${APP_HOME}" /app

if [[ "${ENABLE_SSHD}" == "1" ]]; then
  mkdir -p /var/run/sshd "${APP_HOME}/.ssh"
  touch "${APP_HOME}/.ssh/authorized_keys"

  if [[ -n "${PUBLIC_KEY:-}" ]]; then
    if ! grep -qxF "${PUBLIC_KEY}" "${APP_HOME}/.ssh/authorized_keys"; then
      printf '%s\n' "${PUBLIC_KEY}" >> "${APP_HOME}/.ssh/authorized_keys"
    fi
  fi

  chmod 700 "${APP_HOME}/.ssh"
  chmod 600 "${APP_HOME}/.ssh/authorized_keys"
  chown -R "${APP_USER}:${APP_USER}" "${APP_HOME}/.ssh"

  cat > /etc/ssh/sshd_config.d/99-engine.conf <<EOF
Port ${SSH_PORT}
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
PermitRootLogin no
AllowUsers ${APP_USER}
EOF

  /usr/sbin/sshd
fi

exec su -s /bin/bash -c "/app/entrypoint.sh" "${APP_USER}"
