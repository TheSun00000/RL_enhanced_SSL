import sys
import subprocess

def run_command(arg):
    command = f'squeue | grep "{arg}"'
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout.strip('\n'))
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <argument>")
        sys.exit(1)

    arg = sys.argv[1]
    run_command(arg)