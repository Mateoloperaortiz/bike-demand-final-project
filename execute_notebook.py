"""Ejecuta todas las celdas en un kernel limpio y conserva salidas y vista HTML."""
from pathlib import Path
import asyncio
import sys
import time
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

root = Path(__file__).resolve().parent
path = root / 'model_development.ipynb'
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
started = time.perf_counter()


def report_start(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Celda {cell_index + 1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:95]}', flush=True)


client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                        resources={'metadata': {'path': str(root)}},
                        on_cell_start=report_start)
try:
    client.execute()
except Exception:
    # Conservar el diagnóstico sin fingir ejecución completa.
    nbformat.write(notebook, path)
    raise
nbformat.write(notebook, path)
exporter = HTMLExporter(template_name='lab')
html, _ = exporter.from_notebook_node(notebook)
preview = root / 'tmp' / 'notebook_preview.html'
preview.parent.mkdir(exist_ok=True)
preview.write_text(html, encoding='utf-8')
print(f'Ejecución completa en {time.perf_counter()-started:.1f} s. Vista: {preview}', flush=True)
