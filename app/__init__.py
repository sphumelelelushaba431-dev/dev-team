"""CampusVoice: RAG-powered student complaint management system."""
__version__ = "1.0.0"


def _shim_blocked_sklearn_dll():
    """Windows Smart App Control can block sklearn's NMF extension (_cdnmf_fast),
    which breaks `import sklearn.decomposition` even though we only use TruncatedSVD.
    If that happens, register a stub so the rest of sklearn imports normally."""
    import sys
    import types
    name = "sklearn.decomposition._cdnmf_fast"
    try:
        __import__(name)
    except ImportError as exc:
        if "Application Control" not in str(exc) and "DLL load failed" not in str(exc):
            return
        stub = types.ModuleType(name)

        def _update_cdnmf_fast(*_a, **_kw):
            raise RuntimeError("NMF is unavailable: Windows blocked sklearn's _cdnmf_fast DLL")
        stub._update_cdnmf_fast = _update_cdnmf_fast
        sys.modules[name] = stub


_shim_blocked_sklearn_dll()


def create_app():
    """Lets the Flask CLI find the app: flask --app app <command>"""
    from .main import app
    return app
