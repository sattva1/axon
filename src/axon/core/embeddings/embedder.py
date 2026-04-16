"""Batch embedding pipeline for Axon knowledge graphs.

Takes a :class:`KnowledgeGraph`, generates natural-language descriptions for
each embeddable symbol node, encodes them using *fastembed*, and returns a
list of :class:`NodeEmbedding` objects ready for storage.

Only code-level symbol nodes are embedded.  Structural nodes (Folder,
Community, Process) are deliberately skipped — they lack the semantic
richness that makes embedding worthwhile.
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

from axon.core.embeddings.text import build_class_method_index, generate_text
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel
from axon.core.storage.base import EMBEDDING_DIMENSIONS, NodeEmbedding

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

_model_cache: dict[tuple[str, bool, bool], "TextEmbedding"] = {}
_model_lock = threading.Lock()
_cuda_enabled: bool = False
_coreml_enabled: bool = False


def configure_cuda(enabled: bool) -> None:
    """Enable or disable CUDA for all embedding operations."""
    global _cuda_enabled
    _cuda_enabled = enabled


def configure_coreml(enabled: bool) -> None:
    """Enable or disable CoreML for all embedding operations."""
    global _coreml_enabled
    _coreml_enabled = enabled


def _resolve_cuda() -> bool:
    """Return True if CUDA is enabled via configure_cuda() or AXON_CUDA env var."""
    return _cuda_enabled or os.environ.get(
        "AXON_CUDA", ""
    ).strip() in ("1", "true", "yes")


def _resolve_coreml() -> bool:
    """Return True if CoreML is enabled via configure_coreml() or AXON_COREML env var."""
    return _coreml_enabled or os.environ.get(
        "AXON_COREML", ""
    ).strip() in ("1", "true", "yes")


def _get_model(model_name: str) -> "TextEmbedding":
    cuda = _resolve_cuda()
    coreml = _resolve_coreml()
    if cuda and coreml:
        raise RuntimeError("--cuda and --coreml are mutually exclusive")
    cache_key = (model_name, cuda, coreml)
    cached = _model_cache.get(cache_key)
    if cached is not None:
        return cached
    with _model_lock:
        cached = _model_cache.get(cache_key)
        if cached is not None:
            return cached
        from fastembed import TextEmbedding

        # Cap ONNX threads to avoid saturating all CPU cores.
        # Default to half the available cores (minimum 2).
        max_threads = max(2, os.cpu_count() // 2) if os.cpu_count() else 2
        if cuda:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = TextEmbedding(
                    model_name=model_name, threads=max_threads, cuda=True
                )
            for w in caught:
                if issubclass(w.category, RuntimeWarning) and (
                    "CUDAExecutionProvider" in str(w.message)
                ):
                    raise RuntimeError(
                        "--cuda / AXON_CUDA requested but "
                        "CUDAExecutionProvider failed to initialize.\n"
                        "Install CUDA dependencies:\n"
                        "  pip install onnxruntime-gpu "
                        "nvidia-cublas-cu12 nvidia-cudnn-cu12 "
                        "nvidia-cufft-cu12 nvidia-curand-cu12 "
                        "nvidia-cuda-runtime-cu12\n"
                        "See https://onnxruntime.ai/docs/execution-providers"
                        "/CUDA-ExecutionProvider.html"
                    )
        elif coreml:
            try:
                model = TextEmbedding(
                    model_name=model_name,
                    threads=max_threads,
                    providers=[
                        "CoreMLExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                )
            except ValueError:
                raise RuntimeError(
                    "--coreml / AXON_COREML requested but "
                    "CoreMLExecutionProvider is not available.\n"
                    "Install onnxruntime with CoreML support:\n"
                    "  pip install onnxruntime\n"
                    "(CoreML EP is included by default on macOS ARM builds)"
                )
        else:
            # Explicitly disable CUDA to override fastembed's Device.AUTO
            # default, which would auto-detect and use GPU when
            # onnxruntime-gpu is installed.
            model = TextEmbedding(
                model_name=model_name, threads=max_threads, cuda=False
            )
        _model_cache[cache_key] = model
        return model


def _get_model_cache_clear() -> None:
    """Clear the model cache (used in tests)."""
    with _model_lock:
        _model_cache.clear()


_get_model.cache_clear = _get_model_cache_clear  # type: ignore[attr-defined]


def validate_cuda() -> None:
    """Eagerly initialize the default model to validate CUDA configuration.

    Raises RuntimeError if CUDA was requested but CUDAExecutionProvider
    failed to initialize.
    """
    if not _resolve_cuda():
        return
    _get_model(_DEFAULT_MODEL)


def validate_coreml() -> None:
    """Eagerly initialize the default model to validate CoreML configuration.

    Raises RuntimeError if CoreML was requested but CoreMLExecutionProvider
    is not available.
    """
    if not _resolve_coreml():
        return
    _get_model(_DEFAULT_MODEL)


# Labels worth embedding — skip Folder, Community, Process (structural only).
EMBEDDABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.FILE,
        NodeLabel.FUNCTION,
        NodeLabel.CLASS,
        NodeLabel.METHOD,
        NodeLabel.INTERFACE,
        NodeLabel.TYPE_ALIAS,
        NodeLabel.ENUM,
    }
)

_DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_DEFAULT_DIMENSIONS = EMBEDDING_DIMENSIONS  # 384 via Matryoshka
# Nomic-embed-text-v1.5 has 12 attention heads and 2048-token context.
# With long texts, each batch element's attention matrix is ~192 MB
# (12 * 2048 * 2048 * 4 bytes).  Batch size 8 keeps peak memory under
# ~2 GB for the attention pass, safe for both CPU and 8 GB GPUs.
_DEFAULT_BATCH_SIZE = 8
_MAX_TEXT_CHARS = 8192


def embed_query(
    query: str,
    model_name: str = _DEFAULT_MODEL,
    dimensions: int = _DEFAULT_DIMENSIONS,
) -> list[float] | None:
    """Embed a single query string, returning ``None`` on failure."""
    if not query or not query.strip():
        return None
    try:
        model = _get_model(model_name)
        vec = next(iter(model.query_embed(query)))
        return vec[:dimensions].tolist()
    except Exception:
        logger.warning("embed_query failed", exc_info=True)
        return None


def _embed_node_list(
    nodes: list,
    texts: list[str],
    model_name: str,
    batch_size: int,
    dimensions: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[NodeEmbedding]:
    """Embed a list of nodes with their corresponding texts."""
    if not texts:
        return []

    model = _get_model(model_name)
    total = len(texts)
    results: list[NodeEmbedding] = []

    for i, (node, vector) in enumerate(
        zip(nodes, model.passage_embed(texts, batch_size=batch_size))
    ):
        results.append(
            NodeEmbedding(
                node_id=node.id,
                embedding=vector[:dimensions].tolist(),
            )
        )
        if progress_callback and (i + 1) % batch_size == 0:
            progress_callback(i + 1, total)

    if progress_callback:
        progress_callback(total, total)

    return results


def embed_graph(
    graph: KnowledgeGraph,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    dimensions: int = _DEFAULT_DIMENSIONS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[NodeEmbedding]:
    """Generate embeddings for all embeddable nodes in the graph.

    Uses fastembed's :class:`TextEmbedding` model for batch encoding.
    Each embeddable node is converted to a natural-language description
    via :func:`generate_text`, then embedded in a single batch call.

    Args:
        graph: The knowledge graph whose nodes should be embedded.
        model_name: The fastembed model identifier. Defaults to
            ``"nomic-ai/nomic-embed-text-v1.5"``.
        batch_size: Number of texts to encode per batch. Defaults to 128.
        dimensions: Number of dimensions for Matryoshka truncation.
            Defaults to 384.
        progress_callback: Optional callback receiving (done, total) item
            counts as embedding progresses.

    Returns:
        A list of :class:`NodeEmbedding` instances, one per embeddable node,
        each carrying the node's ID and its embedding vector as a plain
        Python ``list[float]``.
    """
    all_nodes = [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]
    if not all_nodes:
        return []

    class_method_idx = build_class_method_index(graph)

    texts: list[str] = []
    nodes = []
    for node in all_nodes:
        text = generate_text(node, graph, class_method_idx)
        if text and text.strip():
            texts.append(text[:_MAX_TEXT_CHARS])
            nodes.append(node)

    if not texts:
        return []

    return _embed_node_list(
        nodes,
        texts,
        model_name,
        batch_size,
        dimensions,
        progress_callback=progress_callback,
    )


def embed_nodes(
    graph: KnowledgeGraph,
    node_ids: set[str],
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    dimensions: int = _DEFAULT_DIMENSIONS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[NodeEmbedding]:
    """Like :func:`embed_graph`, but only for the given *node_ids*."""
    if not node_ids:
        return []
    nodes = [graph.get_node(nid) for nid in node_ids]
    nodes = [n for n in nodes if n is not None and n.label in EMBEDDABLE_LABELS]
    if not nodes:
        return []

    class_method_idx = build_class_method_index(graph)

    texts: list[str] = []
    valid_nodes = []
    for node in nodes:
        text = generate_text(node, graph, class_method_idx)
        if text and text.strip():
            texts.append(text[:_MAX_TEXT_CHARS])
            valid_nodes.append(node)

    if not texts:
        return []

    return _embed_node_list(
        valid_nodes,
        texts,
        model_name,
        batch_size,
        dimensions,
        progress_callback=progress_callback,
    )
