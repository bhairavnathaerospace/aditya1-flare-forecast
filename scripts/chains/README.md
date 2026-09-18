# Run chains

Each file is a queue that was launched detached (see
`memory/long-runs-detached.md`): a multi-hour sequence of CLI stages that
survives the terminal or chat session that started it. They are kept because
they record exactly what produced each result folder, and they can be re-run.

| Chain | Ran | What it did |
|---|---|---|
| `resume_chain.py` | 2026-09-15 | Restarted the GOES run after it was killed mid cross-validation; re-scored the model with the fixed persistence mask |
| `resume_chain2.py` | 2026-09-15 | Final steps of that run: post-hoc re-scoring, freeze v2, report |
| `seed_chain.py` | 2026-09-16 | Repeated the HEL1OS fusion ablation with TCN seeds 1338–1340 and a linear encoder |
| `day2_chain.py` | 2026-09-16 | Learning curve → PatchTST → recalibration → anchored-flux retrain → fair references |
| `day3_patchtst.py` | 2026-09-16 | Re-ran the PatchTST step that day2 lost to a wrong command name |
| `day3_freeze_v3.py` | 2026-09-16 | Froze the anchored model as v3 |

Launch one the way they were launched (Windows, own process group so Ctrl+C
cannot kill it), from the project root:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32](0x200 -bor 0x08000000) }
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c "python -u scripts\chains\<chain>.py > outputs\archive_goes\reports\<chain>.log 2>&1"'
  CurrentDirectory = (Get-Location).Path; ProcessStartupInformation = $si }
```

`scripts/stay_awake.py <pid> <grace_seconds>` holds a Windows "system required"
power request while such a run is in progress. It changes no power settings.
