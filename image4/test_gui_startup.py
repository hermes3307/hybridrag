#!/usr/bin/env python3
"""
Quick test to check if GUI starts with tabs visible
"""
import subprocess
import time
import os
import signal

print("Starting GUI test...")
print("This will launch the GUI for 5 seconds to verify tabs are visible")

# Start the GUI
proc = subprocess.Popen(['python3', 'image.py'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       cwd='/home/pi/hybridrag/image')

print("GUI launched. Please check if:")
print("  1. All tabs are visible (System Overview, Download Images, Process & Embed, Search Images, Configuration)")
print("  2. Logs appear in the System Overview tab")
print("  3. Initialization messages are shown")
print("\nWaiting 5 seconds...")

time.sleep(5)

# Kill the GUI
proc.terminate()
time.sleep(1)
if proc.poll() is None:
    proc.kill()

stdout, stderr = proc.communicate()

print("\n=== STDOUT ===")
print(stdout.decode('utf-8')[:1000])
print("\n=== STDERR ===")
print(stderr.decode('utf-8')[:1000])

print("\nTest complete. If you saw the GUI with all tabs visible, the fix is working!")
