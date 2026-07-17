import base64
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlmodel import Session

from common.config import OTA_DOWNLOAD_COOLDOWN_SECONDS
from common.db import (
    User,
    get_firmware,
    get_session,
    list_firmware,
)
from common.ratelimit import RateLimit
from api.lib.challenge import require_device_owner
from api.lib.ratelimit import check_limit, record_usage
from api.lib.session import get_current_user

router = APIRouter(prefix="/ota")

FIRMWARE_SIGNATURE_HEADER = "X-Firmware-Signature"


class FirmwareInfo(BaseModel):
    version: str
    interface_version: int
    supported_contracts: list[int]
    size: int
    created_at: datetime


@router.get("/versions/{interface}", response_model=list[FirmwareInfo])
def list_versions(interface: int,
                  session: Session = Depends(get_session),
                  user: User = Depends(get_current_user)):
    return [FirmwareInfo(
        version=fw.version, interface_version=fw.interface_version,
        supported_contracts=fw.supported_contracts, size=fw.size,
        created_at=fw.created_at,
    ) for fw in list_firmware(session, interface)]


@router.get("/download/{interface}/{version}")
def download_firmware(interface: int, version: str,
                      session: Session = Depends(get_session),
                      user: User = Depends(get_current_user)):
    check_limit(RateLimit.ota_download, user.id, str(interface), 1,
                OTA_DOWNLOAD_COOLDOWN_SECONDS)
    require_device_owner(session, user)

    firmware = get_firmware(session, interface, version)
    if firmware is None:
        raise HTTPException(status_code=404,
                            detail=f"No firmware '{version}' for interface {interface}")

    # The per-interface cooldown is spent only once we actually serve an image,
    # so an unknown version above (404) does not count against it.
    try:
        headers = {
            FIRMWARE_SIGNATURE_HEADER: base64.b64encode(firmware.signature).decode(),
            "Content-Disposition":
                f'attachment; filename="somasafe-firmware-{version}.bin"',
        }
        # zstd-compressed as stored; the client decompresses, then verifies the
        # signature (which covers the raw image) before forwarding it to the device.
        return Response(content=firmware.data,
                        media_type="application/octet-stream", headers=headers)
    finally:
        record_usage(RateLimit.ota_download, user.id, str(interface),
                     OTA_DOWNLOAD_COOLDOWN_SECONDS)
