#!/usr/bin/env python3

import os
import paramiko
import sys

print(os.environ.keys())

SCHEDULER_HOST = os.environ.get("SCHEDULER_HOST", None)
if not SCHEDULER_HOST:
    raise ValueError("The variable SCHEDULER_HOST not defined.")
print("SCHEDULER_HOST=%s" % SCHEDULER_HOST)

WORKER_HOSTS = os.environ.get("WORKER_HOSTS", None)
if not WORKER_HOSTS:
    raise ValueError("The variable WORKER_HOSTS not defined.")
print("WORKER_HOSTS=%s" % WORKER_HOSTS)

SSH_WORKINGDIR = os.environ.get("SSH_WORKINGDIR", None)
if not SSH_WORKINGDIR:
    raise ValueError("The variable SSH_WORKINGDIR not defined.")
print("SSH_WORKINGDIR=%s" % SSH_WORKINGDIR)

SSH_USERNAME = os.environ.get("SSH_USERNAME", None)
if not SSH_USERNAME:
    raise ValueError("The variable SSH_USERNAME not defined.")
print("SSH_USERNAME=%s" % SSH_USERNAME)

SSH_PASSWORD = os.environ.get("SSH_PASSWORD", None)

# Start scheduler.
print("Starting scheduler on: %s" % SCHEDULER_HOST)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.load_system_host_keys()
client.connect(SCHEDULER_HOST, username=SSH_USERNAME, password=SSH_PASSWORD)
stdin, stdout, stderr = client.exec_command("bash %s" % os.path.join(SSH_WORKINGDIR, "dev/src/start-dask-scheduler"))
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print(err)
    sys.exit(-1)

# Start the workers.
for WORKER_HOST in WORKER_HOSTS.split(","):
    WORKER_HOST.strip()

    print("Starting worker on: %s" % WORKER_HOST)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.load_system_host_keys()
    client.connect(WORKER_HOST, username=SSH_USERNAME, password=SSH_PASSWORD)
    stdin, stdout, stderr = client.exec_command("bash %s" % os.path.join(SSH_WORKINGDIR, "dev/src/start-dask-worker"))
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print(err)
        sys.exit(-1)
