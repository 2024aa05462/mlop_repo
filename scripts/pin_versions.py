import subprocess
import json
from datetime import datetime

def capture_environment():
    """Capture exact environment state"""

    env_info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': subprocess.check_output(
            ['python', '--version']
        ).decode().strip(),
        'pip_packages': {},
        'conda_packages': {}
    }

    # Capture pip packages
    try:
        pip_list = subprocess.check_output(['pip', 'list', '--format=json'])
        pip_packages = json.loads(pip_list)
        env_info['pip_packages'] = {p['name']: p['version'] for p in pip_packages}
    except Exception as e:
        print(f"Warning: Could not capture pip packages: {e}")

    # Save to file
    with open('environment_snapshot.json', 'w') as f:
        json.dump(env_info, f, indent=2)

    print("[OK] Environment snapshot saved to environment_snapshot.json")
    return env_info

if __name__ == "__main__":
    capture_environment()
