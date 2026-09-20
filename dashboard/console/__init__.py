"""The mission console: two desktop windows over the pipeline, training and alerts.

    common.py         paths (config/project.toml), palette, widgets, machine load
    jobs.py           one detached job at a time (dashboard/job_runner.py)
    pipeline_view.py  Pipeline screen: stage states from outputs/pipeline/state.json
    training.py       Training screen: live.json / history.json of the run being trained
    system.py         Machine panel: GPU, CPU, memory, disk and training throughput
    flarewatch.py     Flare Watch window: the frozen model's alerts, replayed day by day
    app.py            both windows, the action bar, terminal and watchdog

Everything is refreshed once a second; the matplotlib charts redraw only when
the underlying files change.
"""
