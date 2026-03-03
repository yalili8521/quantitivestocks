#!/usr/bin/env python3
"""Find paper trader processes (python main.py trade) and show PID + command.
Run from project root. Option: --stop-intraday to stop the intraday paper trader process.
"""
import os
import re
import subprocess
import sys

def main():
    stop_intraday = "--stop-intraday" in sys.argv
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as e:
        print("Error running wmic:", e)
        sys.exit(1)
    if out.returncode != 0:
        print("wmic failed:", out.stderr or out.stdout)
        sys.exit(1)
    # Parse output: CommandLine and ProcessId columns
    lines = out.stdout.strip().splitlines()
    if not lines:
        print("No python processes found.")
        return
    header = lines[0]
    # WMIC often has trailing spaces; columns are CommandLine, ProcessId
    processes = []
    for line in lines[1:]:
        line = line.rstrip()
        if not line:
            continue
        # Last part is usually PID (digits)
        parts = line.rsplit(None, 1)
        if len(parts) >= 2 and parts[-1].isdigit():
            pid = int(parts[-1])
            cmd = parts[0].strip() if len(parts) == 2 else line
        else:
            cmd = line
            pid = None
        if "main.py" in cmd and "trade" in cmd:
            processes.append((pid, cmd))
    if not processes:
        print("No paper trader (main.py trade) processes found.")
        print("Lock files may be stale. You can delete them from %TEMP%:")
        print("  .paper_trader_intraday.lock")
        return
    print("Paper trader process(es):\n")
    intraday_pid = None
    for pid, cmd in processes:
        short = cmd[:100] + "..." if len(cmd) > 100 else cmd
        print("  PID:", pid)
        print("  Cmd:", short)
        if "intraday" in cmd.lower() or "group" not in cmd:
            intraday_pid = pid  # might be intraday if no other group specified
        print()
    if stop_intraday and processes:
        # Prefer the one that has intraday in args
        pid_to_stop = intraday_pid or processes[0][0]
        print("Stopping PID", pid_to_stop, "...")
        try:
            subprocess.run(["taskkill", "/PID", str(pid_to_stop), "/F"], check=True, timeout=5)
            print("Stopped.")
        except subprocess.CalledProcessError as e:
            print("Failed to stop:", e)
            print("Try running as Administrator or stop the process in Task Manager.")
        except Exception as e:
            print("Failed:", e)


if __name__ == "__main__":
    main()
