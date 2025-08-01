import os, subprocess, pathlib, shlex, sys

def compile_metal_source(src_path: str):
    file_name = os.path.splitext(src_path)[0]
    dir_ = os.path.dirname(src_path)
    metallib_path = os.path.join(dir_, f"{file_name}.metallib")

    cmd = (
        f"xcrun -sdk macosx metal -c {shlex.quote(src_path)} -o - | "
        f"xcrun -sdk macosx metallib - -o {shlex.quote(metallib_path)}"
    )

    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        # e.stdout / e.stderr contain the compiler messages
        print("Metal compilation failed:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        raise

def remove_file(src_path: str):
    os.remove(src_path)