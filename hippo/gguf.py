"""GGUF metadata reading utilities."""

import struct
import hashlib
import logging

from hippo.model_manager import ModelManager

logger = logging.getLogger("hippo")


def _read_gguf_metadata_fast(path, max_keys=None):
    """Read GGUF metadata without loading the model (fast, <1ms).

    Parses the GGUF header directly to extract key-value metadata pairs.
    This avoids the expensive llama_cpp.Llama() constructor which loads
    the entire model into memory just to read metadata.

    GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    """
    GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12

    try:
        with open(path, "rb") as f:
            # Read header: magic(4) + version(4) + n_tensors(8) + n_kv(8)
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                return {}

            version = struct.unpack("<I", f.read(4))[0]
            if version >= 3:
                struct.unpack("<Q", f.read(8))[0]  # n_tensors (uint64)
            else:
                struct.unpack("<I", f.read(4))[0]  # n_tensors (uint32)
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def read_string():
                length = struct.unpack("<Q", f.read(8))[0]
                return f.read(length).decode("utf-8", errors="replace")

            def read_value(vtype):
                if vtype == GGUF_TYPE_UINT8:
                    return struct.unpack("<B", f.read(1))[0]
                elif vtype == GGUF_TYPE_INT8:
                    return struct.unpack("<b", f.read(1))[0]
                elif vtype == GGUF_TYPE_UINT16:
                    return struct.unpack("<H", f.read(2))[0]
                elif vtype == GGUF_TYPE_INT16:
                    return struct.unpack("<h", f.read(2))[0]
                elif vtype == GGUF_TYPE_UINT32:
                    return struct.unpack("<I", f.read(4))[0]
                elif vtype == GGUF_TYPE_INT32:
                    return struct.unpack("<i", f.read(4))[0]
                elif vtype == GGUF_TYPE_FLOAT32:
                    return struct.unpack("<f", f.read(4))[0]
                elif vtype == GGUF_TYPE_BOOL:
                    return struct.unpack("<B", f.read(1))[0] != 0
                elif vtype == GGUF_TYPE_STRING:
                    return read_string()
                elif vtype == GGUF_TYPE_UINT64:
                    return struct.unpack("<Q", f.read(8))[0]
                elif vtype == GGUF_TYPE_INT64:
                    return struct.unpack("<q", f.read(8))[0]
                elif vtype == GGUF_TYPE_FLOAT64:
                    return struct.unpack("<d", f.read(8))[0]
                elif vtype == GGUF_TYPE_ARRAY:
                    elem_type = struct.unpack("<I", f.read(4))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    # Skip array data - we don't need it for metadata lookup
                    elem_sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                    if elem_type == 8:  # string array
                        for _ in range(arr_len):
                            slen = struct.unpack("<Q", f.read(8))[0]
                            f.read(slen)
                    elif elem_type in elem_sizes:
                        f.read(arr_len * elem_sizes[elem_type])
                    return None
                else:
                    return None

            metadata = {}
            for _ in range(n_kv):
                key = read_string()
                vtype = struct.unpack("<I", f.read(4))[0]
                value = read_value(vtype)
                if value is not None:
                    metadata[key] = value
                if max_keys and len(metadata) >= max_keys:
                    break

            return metadata
    except Exception:
        return {}


def _model_info(name: str, manager: ModelManager) -> dict:
    """Build Ollama-style model info dict."""
    model_path = manager._resolve_model_path(name)
    family = "unknown"
    quant = ""
    size_bytes = 0
    if model_path:
        family = manager._detect_family(model_path)
        size_bytes = model_path.stat().st_size
        # Extract quantization from filename (e.g., Q4_K_M)
        fname = model_path.name.upper()
        for q in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K", "F16", "F32"]:
            if q in fname:
                quant = q
                break

    # Quick digest (first 8 bytes of SHA256 of filename, not full file hash)
    digest = hashlib.sha256(name.encode()).hexdigest()[:64]

    return {
        "name": name,
        "model": name,
        "modified_at": "",
        "size": size_bytes,
        "digest": digest,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": "",
            "quantization_level": quant,
        },
    }
