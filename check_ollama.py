import subprocess
import sys

print('PYTHON', sys.executable)
res = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
print('RC', res.returncode)
print('STDOUT_START')
print(res.stdout)
print('STDOUT_END')
print('STDERR_START')
print(res.stderr)
print('STDERR_END')
