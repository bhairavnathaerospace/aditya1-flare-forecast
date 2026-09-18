"""Hold a Windows 'system required' power request while a process runs.

This is the per-process request media players use (SetThreadExecutionState):
it changes no power settings and ends automatically when this script exits.
Runs until the watched process exits, plus a grace period.
"""
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

pid = int(sys.argv[1])
grace_s = float(sys.argv[2]) if len(sys.argv) > 2 else 5400
k32 = ctypes.windll.kernel32


def alive(p: int) -> bool:
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, p)
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == STILL_ACTIVE


k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print(f"holding system-required power request while pid {pid} runs", flush=True)
while alive(pid):
    time.sleep(60)
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print(f"pid {pid} exited; holding {grace_s / 60:.0f} more min for analysis", flush=True)
end = time.time() + grace_s
while time.time() < end:
    time.sleep(60)
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
k32.SetThreadExecutionState(ES_CONTINUOUS)
print("released", flush=True)
