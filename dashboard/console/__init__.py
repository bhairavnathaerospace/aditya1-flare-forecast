"""The mission console: a desktop window over the pipeline, training and alerts.

    common.py         paths (config/project.toml), palette, widgets, file helpers
    jobs.py           one detached job at a time (dashboard/job_runner.py)
    pipeline_view.py  Pipeline tab: stage states from outputs/pipeline/state.json
    training.py       Training tab: live.json / history.json of the run being trained
    flarewatch.py     Flare Watch tab: the frozen model's alerts, replayed day by day
    app.py            the window, action bar, terminal and watchdog
"""
