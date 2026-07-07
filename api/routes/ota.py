import base64
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlmodel import Session

from common.config import OTA_DOWNLOAD_COOLDOWN_SECONDS
from common.db import User, get_firmware, get_session, list_firmware
from api.lib.ratelimit import enforce_cooldown
from .auth import get_current_user
from .model import require_device_owner

router = APIRouter(prefix="/ota")

FIRMWARE_SIGNATURE_HEADER = "X-Firmware-Signature"


class FirmwareInfo(BaseModel):
    """A published firmware build for one BLE interface version. The client
    filters by its own ``BLE_INTERFACE_VERSION`` (the path parameter) and picks
    the newest entry; ``supported_contracts`` tells it which model contract
    versions the build can run."""

    version: str
    interface_version: int
    supported_contracts: list[int]
    size: int
    created_at: datetime


@router.get("/versions/{interface}", response_model=list[FirmwareInfo])
def list_versions(interface: int,
                  session: Session = Depends(get_session),
                  user: User = Depends(get_current_user)):
    """Every firmware build published for a BLE interface version, newest
    first. An interface with no published builds yields an empty list."""
    return [FirmwareInfo(
        version=fw.version, interface_version=fw.interface_version,
        supported_contracts=fw.supported_contracts, size=len(fw.blob),
        created_at=fw.created_at,
    ) for fw in list_firmware(session, interface)]


@router.get("/download/{interface}/{version}")
def download_firmware(interface: int, version: str,
                      session: Session = Depends(get_session),
                      user: User = Depends(get_current_user)):
    """Serve a firmware image. The signature header carries the server's ECDSA
    over the raw image bytes, which the app forwards to the device's BLE OTA
    service for verification against its factory srv_pub."""
    require_device_owner(session, user)
    enforce_cooldown("ota-download", user.id, str(interface),
                     OTA_DOWNLOAD_COOLDOWN_SECONDS)

    firmware = get_firmware(session, interface, version)
    if firmware is None:
        raise HTTPException(status_code=404,
                            detail=f"No firmware '{version}' for interface {interface}")

    headers = {
        "Content-Disposition":
            f'attachment; filename="somasafe-firmware-{version}.bin"',
    }
    if firmware.signature is not None:
        headers[FIRMWARE_SIGNATURE_HEADER] = base64.b64encode(firmware.signature).decode()

    return Response(content=bytes(firmware.blob),
                    media_type="application/octet-stream", headers=headers)
