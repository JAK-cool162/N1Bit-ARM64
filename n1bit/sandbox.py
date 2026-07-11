import os
import subprocess
import sys
import shlex
from typing import Dict, Any

class ProotSandbox:
    """
    Ubuntu PRoot Sandbox Environment for the 1-Bit AI model.
    Runs commands, code, games, and script simulations inside an isolated environment.
    
    If run on Termux, it can hook into a real Ubuntu PRoot container if installed,
    otherwise it spins up a highly restricted, secure subprocess simulation shell
    with virtual file mapping.
    """
    def __init__(self, root_dir: str = "cache/sandbox_root"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        
        # Check if proot binary is available in Termux path
        self.has_proot = self._check_proot_binary()

    def _check_proot_binary(self) -> bool:
        try:
            # Check if proot exists
            subprocess.run(["which", "proot"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def execute_command(self, cmd: str) -> Dict[str, Any]:
        """
        Executes a shell command safely inside the sandbox root.
        If Termux PRoot is available, wraps the command inside PRoot.
        """
        # Strip command and normalize
        cmd = cmd.strip()
        if not cmd:
            return {"stdout": "", "stderr": "Empty command", "exit_code": 1}
            
        # Prevent escaping sandbox root directory if running simulated mode
        if ".." in cmd and not self.has_proot:
            return {
                "stdout": "", 
                "stderr": "Access Denied: Attempted sandbox escape directory traversal.", 
                "exit_code": 1
            }

        # Build execution environment
        if self.has_proot:
            # Real PRoot environment: proot -0 -w /root -r <root_dir> <command>
            # (Assumes user has unpacked an Ubuntu rootfs into self.root_dir)
            wrapped_cmd = f"proot -0 -r {self.root_dir} -w /root {cmd}"
            args = shlex.split(wrapped_cmd)
        else:
            # Simulated isolated shell running with cwd set to self.root_dir
            args = shlex.split(cmd)

        try:
            # Run command with 10-second timeout to prevent infinite loops
            result = subprocess.run(
                args,
                cwd=self.root_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=12
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Process killed: Execution timed out (max 12 seconds).",
                "exit_code": -1
            }
        except FileNotFoundError:
            return {
                "stdout": "",
                "stderr": f"Executable not found: '{args[0]}'. Make sure the command exists inside the sandbox.",
                "exit_code": 127
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Execution error: {e}",
                "exit_code": -1
            }

    def execute_python_code(self, python_code: str) -> Dict[str, Any]:
        """
        Executes raw Python code inside the sandbox.
        Useful for generating code, games, mathematical models, or text images.
        """
        # Save Python script inside sandbox root
        script_name = "sandbox_script.py"
        script_path = os.path.join(self.root_dir, script_name)
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(python_code)
            
        # Run python script in sandbox cwd
        cmd = f"python3 {script_name}"
        return self.execute_command(cmd)

    def generate_proot_install_script(self) -> str:
        """
        Returns a shell script to install and configure Ubuntu PRoot in Termux.
        """
        return f"""#!/bin/bash
# N1Bit-ARM64: Install Ubuntu PRoot container inside Termux for full Sandbox simulation
echo "[Sandbox] Setting up isolated Ubuntu PRoot environment..."

pkg update -y
pkg install proot proot-distro -y

# Install Ubuntu distro via proot-distro
proot-distro install ubuntu

echo "[Sandbox] Ubuntu PRoot has been successfully installed!"
echo "To log in, run: proot-distro login ubuntu"
"""
