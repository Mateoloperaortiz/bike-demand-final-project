"""Comprueba el acceso de un visitante nuevo sin usar cookies de navegador.

Uso: python scripts/check_public_access.py
HTTP y protobuf/WebSocket, con las bibliotecas ya incluidas por Streamlit.
El prefijo /~/+ corresponde al iframe observado de Streamlit Community Cloud.
No es una prueba visual ni un navegador en incógnito; el protocolo puede cambiar.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

import requests
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from tornado.httpclient import HTTPRequest
from tornado.websocket import websocket_connect

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bike_demand.inference import WEATHER_LABELS, load_artifacts, predict_demand, scenario_features


async def read_run(connection) -> dict:
    result = {"widgets": [], "metrics": [], "exceptions": [], "alerts": [], "page_hash": "", "finished": None, "server_environment": {}}
    for _ in range(200):
        data = await asyncio.wait_for(connection.read_message(), timeout=30)
        if data is None:
            raise RuntimeError("WebSocket cerrado antes de terminar el script.")
        message = ForwardMsg()
        message.ParseFromString(data)
        kind = message.WhichOneof("type")
        if kind == "new_session":
            result["page_hash"] = message.new_session.page_script_hash
            environment = message.new_session.initialize.environment_info
            result["server_environment"] = {
                "streamlit_version": environment.streamlit_version,
                "python_version": environment.python_version,
                "server_os": environment.server_os,
            }
        elif kind == "delta" and message.delta.WhichOneof("type") == "new_element":
            element = message.delta.new_element
            element_kind = element.WhichOneof("type")
            if element_kind in ("button", "selectbox", "date_input", "slider"):
                result["widgets"].append((element_kind, getattr(element, element_kind)))
            elif element_kind == "metric":
                result["metrics"].append({"label": element.metric.label, "value": element.metric.body})
            elif element_kind == "exception":
                result["exceptions"].append({"type": element.exception.type, "message": element.exception.message})
            elif element_kind == "alert":
                result["alerts"].append(element.alert.body)
        elif kind == "script_finished":
            result["finished"] = int(message.script_finished)
            return result
    raise RuntimeError("No llegó script_finished dentro del presupuesto de mensajes.")


def form_message(current: dict, scenario: dict) -> BackMsg:
    message = BackMsg()
    message.rerun_script.page_script_hash = current["page_hash"]
    message.rerun_script.query_string = ""
    for kind, widget in current["widgets"]:
        if getattr(widget, "form_id", "") != "scenario_form":
            continue
        state = message.rerun_script.widget_states.widgets.add()
        state.id = widget.id
        if kind == "button":
            state.trigger_value = bool(widget.is_form_submitter)
        elif kind == "date_input":
            state.string_array_value.data.append(scenario["date"].replace("-", "/"))
        elif kind == "slider":
            state.double_array_value.data.append(float(scenario["humidity_pct"]))
        elif kind == "selectbox":
            state.string_value = {
                "Hora de inicio": f"{scenario['hour']:02d}:00",
                "Condición meteorológica": WEATHER_LABELS[scenario["weather"]],
                "Nivel térmico": scenario["thermal_level"],
                "Nivel de viento": scenario["wind_level"],
            }[widget.label]
    return message


def displayed_prediction(run: dict) -> int | None:
    metrics = [metric for metric in run["metrics"] if metric["label"] == "Alquileres estimados durante una hora"]
    return int(metrics[0]["value"].replace(".", "").replace(",", "")) if metrics else None


async def check_websocket(app_base: str, origin: str, headers: dict, model, metadata: dict, report: dict) -> None:
    request = HTTPRequest(app_base.replace("https://", "wss://") + "/_stcore/stream", headers={"Origin": origin, **headers}, connect_timeout=25, request_timeout=25)
    connection = await websocket_connect(request, subprotocols=["streamlit"])
    try:
        initial = BackMsg()
        initial.rerun_script.query_string = ""
        connection.write_message(initial.SerializeToString(), binary=True)
        current = await read_run(connection)
        websocket_report = {
            "connected": True, "subprotocol": connection.selected_subprotocol,
            "protocol_api": "streamlit.proto.BackMsg_pb2.BackMsg y streamlit.proto.ForwardMsg_pb2.ForwardMsg",
            "streamlit_client_version": version("streamlit"),
            "server_environment": current["server_environment"],
            "initial_script_finished_status": current["finished"],
            "initial_exception_count": len(current["exceptions"]),
            "form_labels": [widget.label for _, widget in current["widgets"] if getattr(widget, "form_id", "") == "scenario_form"],
            "valid_scenarios": [],
        }
        report["websocket"] = websocket_report
        if current["exceptions"] or current["finished"] != 0:
            raise RuntimeError("La carga inicial del script no terminó correctamente.")
        for example in metadata["examples"]:
            scenario = {key: example[key] for key in ("date", "hour", "weather", "thermal_level", "humidity_pct", "wind_level")}
            connection.write_message(form_message(current, scenario).SerializeToString(), binary=True)
            current = await read_run(connection)
            expected = float(predict_demand(model, scenario_features(**scenario, metadata=metadata))[0])
            actual = displayed_prediction(current)
            matched = current["finished"] == 0 and not current["exceptions"] and actual == round(expected)
            websocket_report["valid_scenarios"].append({
                "name": example["name"], "scenario": scenario, "remote_displayed_prediction": actual,
                "local_prediction": expected, "local_rounded_prediction": round(expected),
                "script_finished_status": current["finished"], "exception_count": len(current["exceptions"]), "matched": matched,
            })
            print(f"{example['name']}: remoto {actual}, referencia {round(expected)}; coincide={matched}")
            if not matched:
                raise RuntimeError("Una predicción válida no coincide con el artefacto local.")
        invalid = scenario.copy()
        invalid["date"] = "2013-01-01"
        connection.write_message(form_message(current, invalid).SerializeToString(), binary=True)
        current = await read_run(connection)
        no_prediction = displayed_prediction(current) is None
        validation_alerts = [body for body in current["alerts"] if "fecha" in body.lower() and "2011" in body and "2012" in body]
        graceful = current["finished"] == 0 and not current["exceptions"] and bool(validation_alerts) and no_prediction
        websocket_report["invalid_date_test"] = {
            "date_sent": invalid["date"], "method": "WidgetState directo, evitando la validación frontal del navegador.",
            "script_finished_status": current["finished"], "no_prediction_outside_supported_period": no_prediction,
            "exceptions": current["exceptions"], "alerts": current["alerts"], "date_validation_alerts": validation_alerts,
            "graceful_validation": graceful,
        }
        recovery = {key: metadata["examples"][0][key] for key in ("date", "hour", "weather", "thermal_level", "humidity_pct", "wind_level")}
        connection.write_message(form_message(current, recovery).SerializeToString(), binary=True)
        recovered = await read_run(connection)
        expected = round(float(predict_demand(model, scenario_features(**recovery, metadata=metadata))[0]))
        recovery_passed = recovered["finished"] == 0 and not recovered["exceptions"] and displayed_prediction(recovered) == expected
        websocket_report["recovery_after_invalid_input"] = {
            "remote_displayed_prediction": displayed_prediction(recovered), "expected_prediction": expected,
            "exception_count": len(recovered["exceptions"]), "passed": recovery_passed,
        }
        report["anonymous_access_and_valid_predictions_passed"] = True
        report["invalid_date_graceful_handling_passed"] = graceful
        report["passed"] = graceful and recovery_passed and report["http"]["health_body"] == "ok" and all(report["http"][key] == 200 for key in ("landing_status", "app_iframe_status", "health_status"))
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://bicihora-mateolopera.streamlit.app/")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "reportes/public_access_check.json")
    arguments = parser.parse_args()
    origin = arguments.url.rstrip("/")
    app_base = origin + "/~/+"
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "app_url": origin + "/",
        "method": "Cliente HTTP y WebSocket independiente; no automatiza un navegador.",
        "initial_state": {"cookies": 0, "credentials_provided": False, "browser_cookies_accessed": False, "external_tokens_used": False},
        "limitations": [
            "Esta comprobación técnica no es una prueba de navegador en incógnito.",
            "La sesión comienza sin cookies ni credenciales y conserva solo cookies emitidas por la puerta de visitante de Streamlit Cloud. No significa ausencia de cookies durante todo el flujo.",
            "Valida protocolo, resultados y recuperación; no valida diseño visual ni interacción de un navegador.",
            "El prefijo /~/+ es el iframe observado de esta app en Streamlit Community Cloud; ese protocolo interno puede cambiar.",
        ],
    }
    session = requests.Session()
    session.trust_env = False
    try:
        model, metadata = load_artifacts(PROJECT_DIR / "artifacts")
        landing = session.get(origin + "/", timeout=30)
        app = session.get(app_base + "/", timeout=30)
        health = session.get(app_base + "/_stcore/health", timeout=30)
        responses = landing.history + [landing]
        initial_cookie_sent = "Cookie" in responses[0].request.headers
        authorization_sent = any("Authorization" in response.request.headers for response in responses)
        if initial_cookie_sent or authorization_sent:
            raise RuntimeError("La sesión inicial debe carecer de cookies y credenciales.")
        report["http"] = {
            "initial_request_cookie_header": initial_cookie_sent, "authorization_header_sent": authorization_sent,
            "anonymous_redirects": [{"status": response.status_code, "host": urlparse(response.url).netloc, "path": urlparse(response.url).path} for response in landing.history],
            "landing_status": landing.status_code, "app_iframe_status": app.status_code,
            "health_status": health.status_code, "health_body": health.text[:20],
            "cookies_during_session": bool(session.cookies), "fresh_session_cookie_count": len(session.cookies),
            "cookie_source": "Solo respuestas del flujo nuevo de visitante; valores no guardados.",
        }
        # requests selecciona por dominio/path solo las cookies del host de destino.
        prepared = session.prepare_request(requests.Request("GET", app_base + "/_stcore/stream"))
        cookie_header = prepared.headers.get("Cookie")
        headers = {"Cookie": cookie_header} if cookie_header else {}
        asyncio.run(check_websocket(app_base, origin, headers, model, metadata, report))
    except Exception as error:
        report["passed"] = False
        report["failure"] = {"type": type(error).__name__, "message": "No se completó la verificación. Revisar disponibilidad, protocolo y entorno local."}
        print(f"Verificación incompleta: {type(error).__name__}.", file=sys.stderr)
    finally:
        session.close()
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Reporte: {arguments.output}; passed={report.get('passed', False)}")
    return 0 if report.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
