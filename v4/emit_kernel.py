
from __future__ import annotations

from pathlib import Path

from compiler import compile_program, write_python_kernel
from spec import AUTORESEARCH_PROGRAM


def main(output: str = "compiled_kernel.py") -> None:
    compiled = compile_program(AUTORESEARCH_PROGRAM)
    output_path = Path(output)
    write_python_kernel(compiled, str(output_path))
    print(f"Wrote task-family residual kernel to {output_path}")


if __name__ == "__main__":
    main()
