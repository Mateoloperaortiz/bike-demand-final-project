"""Compone un video de 120 s con capturas reales, sin alterar sus datos.

Uso: python create_demo_video.py
Modo público (solo con URL y capturas previamente verificadas):
    python create_demo_video.py --public-url https://nombre.streamlit.app
No graba el navegador, no descarga imágenes y no vuelve a entrenar el modelo.
Los modos local y público guardan archivos separados para conservar la evidencia.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageOps

from bike_demand.inference import load_artifacts, predict_demand, scenario_features


ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "media"
WIDTH, HEIGHT, FPS = 1920, 1080, 24
INK = "#173d32"
GREEN = "#255e4b"
MUTED = "#62776a"
CREAM = "#f7f8f0"
PALE = "#edf1e5"
LINE = "#dce4d8"
GOLD = "#8b621f"


@dataclass(frozen=True)
class Scene:
    slug: str
    duration: int
    title: str
    subtitle: str
    kind: str
    image: str
    caption: str
    example_index: int | None = None


SCENES = [
    Scene(
        "01_intro", 12, "Cada hora tiene su ritmo.",
        "BiciHora · demanda de bicicletas", "intro", "media/app_laboral.png",
        "BiciHora estima los alquileres de una hora en todo el sistema Capital Bikeshare. "
        "Exploramos escenarios históricos de Washington, de 2011 a 2012.",
    ),
    Scene(
        "02_laboral", 18, "Mañana laboral", "Escenario 1 de 3", "example", "media/app_laboral.png",
        "Seleccionamos Mañana laboral y pulsamos Estimar demanda. "
        "La captura muestra las condiciones y la estimación agregada de una hora.", 0,
    ),
    Scene(
        "03_sabado", 16, "Sábado cálido", "Escenario 2 de 3", "example", "media/app_sabado.png",
        "Cambiamos a Sábado cálido. La fecha, la hora y las condiciones del clima cambian juntas: "
        "estas diferencias describen escenarios, no un efecto causal aislado.", 1,
    ),
    Scene(
        "04_lluvia", 16, "Tarde con lluvia", "Escenario 3 de 3", "example", "media/app_lluvia.png",
        "El tercer ejemplo usa lluvia o nieve, humedad alta y viento alto. "
        "La app conserva el contexto del último escenario calculado.", 2,
    ),
    Scene(
        "05_metodo", 20, "Aprender sin mirar el futuro", "Notebook compatible con Colab · CPU", "method", "figuras/08_comparacion_modelos.png",
        "Los datos se separan por fecha. Ridge, Random Forest y Gradient Boosting se comparan "
        "con validación futura; la prueba final no participa en la selección.",
    ),
    Scene(
        "06_resultados", 20, "Evaluar en horas nuevas", "Prueba final · octubre a diciembre de 2012", "metrics", "figuras/10_errores_prueba.png",
        "En 2.168 horas de prueba, el error absoluto medio fue 52,62 alquileres por hora, "
        "frente a 158,81 de la mediana. El MAE mide error general, no un intervalo individual.",
    ),
    Scene(
        "07_limites", 18, "El promedio no cuenta toda la historia", "Límites y siguiente paso", "limits", "figuras/11_error_horario.png",
        "A las 17:00 el error medio es mayor. El prototipo representa 2011–2012: "
        "una operación actual necesita datos recientes, otra validación y seguimiento del error.",
    ),
]


def public_url_argument(value):
    """Requiere una URL HTTPS pública sin credenciales; no comprueba el alojamiento."""
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise argparse.ArgumentTypeError("La URL pública debe ser HTTPS y no incluir credenciales.")
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("Usa la URL principal de la app, sin parámetros ni fragmentos.")
    return value.rstrip("/")


def selected_scenes(public_url=None):
    """El modo público exige otros archivos; nunca reutiliza las capturas locales."""
    if public_url is None:
        return SCENES
    scenes = []
    for scene in SCENES:
        if scene.image.startswith("media/app_"):
            scene = replace(scene, image=scene.image.replace("media/app_", "media/public_app_", 1))
        if scene.kind == "intro":
            scene = replace(
                scene, subtitle="BiciHora · app pública",
                caption=scene.caption + " Esta demostración es un montaje de capturas reales de la app pública.",
            )
        elif scene.kind == "method":
            scene = replace(scene, subtitle="Notebook ejecutado en Colab · CPU")
        scenes.append(scene)
    return scenes


def font(size: int, bold: bool = False):
    candidates = (
        ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else
        ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    raise FileNotFoundError("Instala una fuente Segoe UI o DejaVu Sans para componer los títulos.")


def text_block(draw, text, xy, width, size=30, color=INK, bold=False, spacing=9):
    """Ajusta texto externo a la captura y devuelve el borde inferior."""
    typeface = font(size, bold)
    x, y = xy
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=typeface) > width and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    line_height = size + spacing
    for line in lines:
        draw.text((x, y), line, font=typeface, fill=color)
        y += line_height
    return y


def number(value, digits=2):
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def image_panel(canvas: Image.Image, source: Path, crop=None, box=(675, 245, 1840, 895)):
    """Solo encuadre, escala proporcional y marco; no modifica contenido de la imagen."""
    with Image.open(source) as original:
        image = original.convert("RGB")
    if crop is not None:
        image = image.crop(crop)
    left, top, right, bottom = box
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=18, fill="#fffef8", outline=LINE, width=2)
    inset = 12
    contained = ImageOps.contain(image, (right - left - 2 * inset, bottom - top - 2 * inset), Image.Resampling.LANCZOS)
    x = left + (right - left - contained.width) // 2
    y = top + (bottom - top - contained.height) // 2
    canvas.paste(contained, (x, y))


def base_scene(scene, index, start, end, public_url=None):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), CREAM)
    draw = ImageDraw.Draw(canvas)
    draw.text((78, 42), "BiciHora", font=font(38, True), fill=INK)
    status = "App pública · capturas verificadas" if public_url else "Demostración local · pendiente de publicación"
    draw.rounded_rectangle((1000, 40, 1840, 91), radius=24, fill=PALE if public_url else "#f2ead7")
    draw.text((1020, 49), status, font=font(27, True), fill=GREEN if public_url else GOLD)
    draw.line((78, 114, 1840, 114), fill=LINE, width=2)
    draw.text((80, 153), scene.subtitle.upper(), font=font(22, True), fill=MUTED)
    title_size = 50 if scene.kind not in ("method", "limits") else 46
    title_end = text_block(draw, scene.title, (78, 200), 540, title_size, bold=True, spacing=6)
    draw.line((78, 999, 1840, 999), fill=LINE, width=2)
    footer = (
        f"Montaje de capturas reales de la app pública · {public_url}"
        if public_url else "Video de capturas del prototipo · sin grabación del alojamiento público"
    )
    text_block(draw, footer, (78, 1022), 1430, 22, color=MUTED, spacing=3)
    draw.text((1535, 1022), f"{index + 1:02d} / {len(SCENES):02d}   ·   {start:03d}–{end:03d} s", font=font(22), fill=MUTED)
    return canvas, draw, title_end


def compose(scene, index, start, model, metadata, summary, public_url=None):
    canvas, draw, title_end = base_scene(scene, index, start, start + scene.duration, public_url)
    source = ROOT / scene.image
    if not source.exists():
        raise FileNotFoundError(f"Falta la captura o figura real: {source}")

    if scene.kind == "intro":
        y = text_block(draw, "Simula el total de alquileres durante una hora.", (80, title_end + 30), 515, 31, color=MUTED)
        y = text_block(draw, "Washington · 2011–2012", (80, y + 25), 520, 29, bold=True)
        y = text_block(draw, "17.379 observaciones horarias\nFuente pública: UCI Bike Sharing", (80, y + 30), 515, 27, color=MUTED)
        text_block(draw, "Calendario + clima → demanda", (80, y + 36), 520, 29, bold=True)
        image_panel(canvas, source, crop=(765, 85, 1800, 605))
        note = (
            "Captura real de la app pública. Los ejemplos se calculan con el modelo guardado."
            if public_url else "Vista real del prototipo local. Los ejemplos se calculan con el modelo guardado."
        )
        text_block(draw, note, (700, 918), 1100, 23, color=MUTED, spacing=5)

    elif scene.kind == "example":
        example = metadata["examples"][scene.example_index]
        inputs = {key: example[key] for key in ("date", "hour", "weather", "thermal_level", "humidity_pct", "wind_level")}
        prediction = float(predict_demand(model, scenario_features(**inputs, metadata=metadata))[0])
        date_label = example["date"].split("-")
        date_label = "/".join(reversed(date_label))
        y = text_block(draw, f"{date_label} · {example['hour']:02d}:00", (80, title_end + 35), 520, 29, bold=True)
        weather = "Despejado" if example["weather"] == 1 else "Lluvia o nieve"
        y = text_block(draw, f"{weather} · {example['thermal_level'].lower()}\nHumedad {example['humidity_pct']} %\nViento {example['wind_level'].lower()}", (80, y + 20), 520, 28, color=MUTED, spacing=15)
        draw.rounded_rectangle((78, y + 35, 590, y + 228), radius=18, fill=GREEN)
        draw.text((104, y + 51), f"{round(prediction):,}".replace(",", "."), font=font(88, True), fill="#fbfff3")
        draw.text((110, y + 163), "alquileres estimados en una hora", font=font(24), fill="#fbfff3")
        text_block(draw, "Todo el sistema Capital Bikeshare", (80, y + 250), 530, 25, color=MUTED)
        image_panel(canvas, source, crop=(760, 500, 1800, 1215), box=(675, 225, 1840, 903))
        text_block(draw, "Captura posterior a Estimar demanda. Se muestran condiciones y resultado del mismo escenario.", (700, 925), 1115, 23, color=MUTED, spacing=4)

    elif scene.kind == "method":
        y = title_end + 28
        periods = [
            ("Entrenamiento", "Ene. 2011–jun. 2012", "13.003 horas"),
            ("Validación", "Jul.–sep. 2012", "2.208 horas"),
            ("Prueba final", "Oct.–dic. 2012", "2.168 horas"),
        ]
        for label, period, rows in periods:
            draw.rounded_rectangle((78, y, 590, y + 126), radius=14, fill=PALE, outline=LINE, width=1)
            draw.text((100, y + 12), label, font=font(27, True), fill=INK)
            draw.text((100, y + 48), period, font=font(26), fill=MUTED)
            draw.text((100, y + 84), rows, font=font(23), fill=MUTED)
            y += 142
        text_block(draw, "Selección: menor MAE en validación. La prueba se abre al final.", (80, y + 10), 515, 25, color=MUTED)
        image_panel(canvas, source, box=(675, 255, 1840, 836))
        text_block(draw, "Comparación en validación: Ridge, Random Forest y Gradient Boosting frente a la mediana.", (700, 866), 1110, 26, color=MUTED, spacing=7)
        text_block(draw, "Pipelines y cinco bloques temporales expansivos evitan ajustar transformaciones con datos futuros.", (700, 932), 1110, 23, color=MUTED, spacing=4)

    elif scene.kind == "metrics":
        metrics = summary["test_metrics"]
        y = title_end + 30
        draw.text((80, y), number(metrics["mae"]), font=font(94, True), fill=GREEN)
        y += 122
        y = text_block(draw, "MAE · alquileres por hora", (82, y), 520, 28, bold=True)
        y = text_block(draw, f"Mediana de referencia: {number(summary['baseline_test_metrics']['mae'])}\nReducción del MAE: {number(summary['mae_reduction_vs_baseline_pct'])} %", (82, y + 28), 510, 26, color=MUTED, spacing=15)
        y = text_block(draw, f"RMSE: {number(metrics['rmse'])}\nR²: {number(metrics['r2'], 3)}", (82, y + 38), 510, 29, bold=True, spacing=16)
        text_block(draw, "MAE resume el error general. No es un intervalo de confianza individual.", (82, y + 33), 515, 26, color=MUTED)
        image_panel(canvas, source, box=(675, 245, 1840, 811))
        text_block(draw, "Modelo final: Gradient Boosting con histogramas", (700, 844), 1110, 28, bold=True)
        text_block(draw, "2.168 horas no vistas. La dispersión muestra errores que una sola cifra puede ocultar.", (700, 891), 1110, 26, color=MUTED, spacing=6)

    elif scene.kind == "limits":
        y = title_end + 25
        y = text_block(draw, "Hora pico: 17:00", (80, y), 520, 30, bold=True)
        y = text_block(draw, "MAE de 134,21 alquileres/hora", (80, y + 18), 520, 28, color=MUTED)
        y = text_block(draw, "El prototipo describe 2011–2012.", (80, y + 38), 520, 30, bold=True)
        y = text_block(draw, "Para operar hoy:\ndatos recientes, nueva validación y seguimiento del error.", (80, y + 22), 520, 28, color=MUTED, spacing=12)
        next_step = (
            "Siguiente paso: seguimiento y actualización con datos recientes."
            if public_url else "Siguiente paso: publicar y verificar el acceso público."
        )
        text_block(draw, next_step, (80, y + 32), 520, 28, bold=True)
        image_panel(canvas, source, box=(675, 245, 1840, 811))
        text_block(draw, "El error cambia con el horario; no basta con mirar el promedio.", (700, 844), 1110, 28, bold=True)
        text_block(draw, "Reajuste: revisar mensualmente el error y las entradas; investigar un aumento del MAE mayor al 20 % durante dos meses. Umbral operativo inicial.", (700, 900), 1110, 25, color=MUTED, spacing=7)
    else:
        raise ValueError(scene.kind)
    return canvas


def subtitle_timestamp(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remainder:02d},000"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-url", type=public_url_argument,
        help="URL de la app pública ya verificada; requiere media/public_app_{laboral,sabado,lluvia}.png.",
    )
    arguments = parser.parse_args()
    public_url = arguments.public_url
    scenes = selected_scenes(public_url)
    assert sum(scene.duration for scene in scenes) == 120
    missing = sorted({scene.image for scene in scenes if not (ROOT / scene.image).is_file()})
    if missing:
        parser.error("Faltan capturas o figuras reales: " + ", ".join(missing))
    prefix = "demo_public" if public_url else "demo_local"
    MEDIA.mkdir(exist_ok=True)
    frames = MEDIA / ("demo_public_frames" if public_url else "demo_frames")
    frames.mkdir(exist_ok=True)
    model, metadata = load_artifacts(ROOT / "artifacts")
    summary = json.loads((ROOT / "reportes/training_summary.json").read_text(encoding="utf-8"))
    concat_lines = ["ffconcat version 1.0"]
    subtitles = []
    scene_records = []
    start = 0
    for index, scene in enumerate(scenes):
        output = frames / f"{scene.slug}.png"
        compose(scene, index, start, model, metadata, summary, public_url).save(output)
        concat_lines.extend([f"file '{output.name}'", f"duration {scene.duration}"])
        end = start + scene.duration
        subtitles.append(f"{index + 1}\n{subtitle_timestamp(start)} --> {subtitle_timestamp(end)}\n{scene.caption}\n")
        source = ROOT / scene.image
        scene_records.append({
            "scene": scene.slug, "start_seconds": start, "end_seconds": end,
            "source": scene.image, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "caption": scene.caption,
        })
        start = end
    concat_lines.append(f"file '{scenes[-1].slug}.png'")
    concat_path = frames / "scenes.ffconcat"
    concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    (MEDIA / f"{prefix}_120s.srt").write_text("\n".join(subtitles), encoding="utf-8")
    movie = MEDIA / f"{prefix}_120s.mp4"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-vf", f"fps={FPS}", "-frames:v", str(120 * FPS), "-t", "120",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(movie),
    ]
    subprocess.run(command, check=True)
    stream = imageio_ffmpeg.read_frames(str(movie))
    video_metadata = next(stream)
    stream.close()
    frame_count, seconds = imageio_ffmpeg.count_frames_and_secs(str(movie))
    assert video_metadata["codec"] == "h264", video_metadata
    assert tuple(video_metadata["source_size"]) == (WIDTH, HEIGHT), video_metadata
    assert frame_count == 120 * FPS, frame_count
    assert abs(seconds - 120.0) <= 0.001, seconds
    assert abs(video_metadata["duration"] - 120.0) <= 0.001, video_metadata
    verification = {
        "file": f"media/{prefix}_120s.mp4", "duration_seconds": seconds,
        "frames": frame_count, "fps": FPS, "width": WIDTH, "height": HEIGHT,
        "codec": video_metadata["codec"], "audio": False,
        "status": "App pública · capturas verificadas" if public_url else "Demostración local · pendiente de publicación",
        "kind": "Montaje de capturas reales de la app pública" if public_url else "Montaje de capturas reales del prototipo, con encuadre y texto externo",
        "public_url": public_url,
        "scenes": scene_records,
    }
    (MEDIA / f"{prefix}_metadata.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in verification.items() if key != "scenes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
