from datetime import date


def build_domain(name: str, today: date | None = None):
    if name == "clinic":
        from .clinic.domain import ClinicDomain

        return ClinicDomain(today=today)
    raise ValueError(f"Неизвестный домен: {name}")
