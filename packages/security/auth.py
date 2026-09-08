"""
Authentication and Role-Based Access Control (RBAC) for AURA.
Uses JWT with role validation and case-level authorization boundaries.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from packages.common.config import get_settings


class UserRole(str, Enum):
    ANALYST = "analyst"
    INVESTIGATOR = "investigator"
    FORENSIC_EXAMINER = "forensic_examiner"
    ADMIN = "admin"
    SYSTEM = "system"


class TokenData(BaseModel):
    subject: str = Field(description="Pseudonymous operator ID or username")
    role: UserRole
    case_ids: list[str] = Field(default_factory=list, description="Authorized case IDs")
    exp: datetime | None = None


security_bearer = HTTPBearer(auto_error=False)


def create_access_token(
    subject: str,
    role: UserRole | str,
    case_ids: list[str] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.JWT_EXPIRE_MINUTES))
    
    role_val = role.value if isinstance(role, UserRole) else str(role)
    payload = {
        "sub": subject,
        "role": role_val,
        "case_ids": case_ids or [],
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "iss": "aura-auth",
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            issuer="aura-auth",
        )
        subject: str | None = payload.get("sub")
        role_str: str | None = payload.get("role")
        case_ids: list[str] = payload.get("case_ids", [])
        exp_ts: int | None = payload.get("exp")
        
        if subject is None or role_str is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing sub or role",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        exp_dt = datetime.fromtimestamp(exp_ts, tz=timezone.utc) if exp_ts else None
        return TokenData(
            subject=subject,
            role=UserRole(role_str),
            case_ids=case_ids,
            exp=exp_dt,
        )
    except (JWTError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(err)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_bearer),
) -> TokenData:
    if not credentials or not credentials.credentials:
        # In development mode, allow anonymous operator if configured
        settings = get_settings()
        if settings.ENVIRONMENT == "development":
            return TokenData(
                subject="dev-analyst-local",
                role=UserRole.ADMIN,
                case_ids=["*"],
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials)


def require_role(*allowed_roles: UserRole | str) -> Callable:
    """Dependency factory that checks whether the user has at least one of the allowed roles."""
    normalized = [r.value if isinstance(r, UserRole) else str(r) for r in allowed_roles]

    async def role_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if current_user.role.value not in normalized and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted for role '{current_user.role.value}'. Required: {normalized}",
            )
        return current_user

    return role_checker
