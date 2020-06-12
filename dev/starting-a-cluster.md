# How to start a Dask Cluster

## Prepare Git Workspace (only once)

1) Create a workspace folder and `git clone` the repo into it.
2) Run `PIPENV_VENV_IN_PROJECT=true pipenv install`

## Start/Stop a Dask Cluster (each time you need machines for experiments)

The dask cluster is made up of one scheduler process and on or more worker processes. With the following procedure, when we start them they will run in the background. Therefore, they don't require an active SSH session to stay alive. When we are done using them, we can stop them with the stopping procedure.

### Start the Cluster

1) Deterimne which machines to use as workers, and which one of them will be the scheduler.
2) SSH into the scheduler machine. Go to the workspace directory.
3) Run `./dev/src/start-dask-scheduler`. It will start in the background.
    * To see the log, run `tail -f /tmp/dask-scheduler.log`
    * To see the web dashboard, the URL is: `http://{SCHEDULER_HOSTNAME}:8787`, where should be replaced with the machine name that hosts that scheduler process
4) SSH into every machine that will be the worker (this includes the scheduler machine). Go to the workspace directory.
5) Run `./dev/src/start-dask-worker`. It will start in the background.
    * To see the log, run `tail -f /tmp/dask-worker.log`
6) In the web dashboard, in the "Workers" tab, verify that the workers have joined the cluster. If yes, the cluster is ready.

### Stop the Cluster

1) SSH into every machine that will be the worker (this includes the scheduler machine). Go to the workspace directory.
2) Run `./dev/src/stop-dask-worker`. It will stop the worker process.
3) SSH into the scheduler machine. Go to the workspace directory.
4) Run `./dev/src/stop-dask-scheduler`. It will stop the scheduler process. The cluster should now be fully stopped.

## Run the Experiment on the Cluster

1) Log onto the same machine where the scheduler is running.
2) Go to the workspace directory and run `pipenv shell`
3) Start the experiment script with the option `cluster="{SCHEDULER_HOSTNAME}:8786"` where `{SCHEDULER_HOSTNAME}` should be replaced with the machine name that hosts the scheduler process, and `8786` is the default port.
