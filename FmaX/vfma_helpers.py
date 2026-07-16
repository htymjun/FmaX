"""Python helpers for FmaX/vfma.f90.fypp's `vfma` macro.

fypp has no directive for embedding multi-statement Python code (no
`#:python`/`#:endpython`, confirmed absent through the latest release, 3.2) --
its only way to run arbitrary Python logic is `-m MODULE`, which imports a
real module and exposes its names to the template. This module is that
import target: invoke fypp as `fypp -M FmaX -m vfma_helpers ...`.
"""
import re


def clean_var_name(base, idx_str):
    # Remove invalid characters (+, -, *, /, ,) to form a valid Fortran variable name.
    # Example: 'i-1' -> 'i_m1', 'i+2' -> 'i_p2', 'i-1,j,k' -> 'i_m1_j_k'
    clean = idx_str.replace(' ', '')
    clean = clean.replace('-', '_m').replace('+', '_p')
    clean = clean.replace('*', '_mul').replace('/', '_div')
    clean = clean.replace(',', '_')
    return f"reg_{base}_{clean}"


def parse_arg(s, n):
    # Parse whether an argument is an array slice like 'a(start:end)' or a scalar like '9.d0'
    s = s.strip()
    if '(' in s and ')' in s:
        base = s.split('(', 1)[0].strip()
        inside = s.split('(', 1)[1].rsplit(')', 1)[0].strip()
        # Split into dimensions on top-level commas (no call site nests parens here).
        dims = [d.strip() for d in inside.split(',')]
        first = dims[0]
        trailing = dims[1:]

        def with_trailing(idx0):
            return ",".join([idx0] + trailing) if trailing else idx0

        if ':' in first:
            # Slice representation (e.g., 'i-1:i+1')
            parts = first.split(':')
            start_str = parts[0].strip()

            # Generate N indices starting from the start index (e.g., 'i-1', 'i', 'i+1')
            match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*([+-])\s*(\d+))?$', start_str)
            if match:
                var, op, offset = match.groups()
                offset = 0 if op is None else (int(offset) if op == '+' else -int(offset))
                idx_list = []
                for k in range(n):
                    cur_off = offset + k
                    if cur_off < 0:
                        idx_list.append(with_trailing(f"{var}-{abs(cur_off)}"))
                    elif cur_off > 0:
                        idx_list.append(with_trailing(f"{var}+{cur_off}"))
                    else:
                        idx_list.append(with_trailing(f"{var}"))
                return {"type": "array", "base": base, "indices": idx_list}
            else:
                # Numeric-only start index (e.g., '1' -> ['1', '2', '3'])
                if start_str.isdigit():
                    start_val = int(start_str)
                    return {"type": "array", "base": base, "indices": [with_trailing(str(start_val + k)) for k in range(n)]}
        else:
            # Single element access (e.g., 'a(i)')
            return {"type": "scalar", "expr": s}
    # No parentheses (literal constants or pure scalar variables)
    return {"type": "scalar", "expr": s}


def compute_loads(infos):
    # Detect memory-space sharing (aliasing) and deduplicate loads.
    # Create a dictionary structure: { 'array_name': { 'index1', 'index2', ... } }
    loads = {}
    for info in infos:
        if info["type"] == "array":
            base = info["base"]
            if base not in loads:
                loads[base] = set()
            for idx in info["indices"]:
                loads[base].add(idx)
    return loads


def resolve_terms(a_info, b_info, c_info, res_info, k):
    # Use register variable if argument is an array, otherwise keep the literal scalar
    a_term = clean_var_name(a_info["base"], a_info["indices"][k]) if a_info["type"] == "array" else a_info["expr"]
    b_term = clean_var_name(b_info["base"], b_info["indices"][k]) if b_info["type"] == "array" else b_info["expr"]
    c_term = clean_var_name(c_info["base"], c_info["indices"][k]) if c_info["type"] == "array" else c_info["expr"]

    # Define target variable for storing the output result
    res_term = f"{res_info['base']}({res_info['indices'][k]})" if res_info["type"] == "array" else res_info["expr"]
    return a_term, b_term, c_term, res_term
