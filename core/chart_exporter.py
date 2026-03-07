import io
import plotly.graph_objects as go


def fig_to_bytes(fig: go.Figure, width: int = 900, height: int = 450) -> bytes:
    """
    Convert a Plotly figure to PNG bytes for embedding in PDF.
    Requires kaleido: pip install kaleido
    """
    try:
        img_bytes = fig.to_image(
            format="png",
            width=width,
            height=height,
            scale=2,
        )
        return img_bytes
    except Exception as e:
        raise RuntimeError(
            f"Chart export failed: {e}. "
            f"Make sure kaleido is installed: pip install kaleido"
        )


def fig_to_buffer(fig: go.Figure, width: int = 900, height: int = 450) -> io.BytesIO:
    """Return chart as BytesIO buffer."""
    buf = io.BytesIO(fig_to_bytes(fig, width, height))
    buf.seek(0)
    return buf
