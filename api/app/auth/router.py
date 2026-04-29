from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from .schemas import RegisterRequest, LoginRequest, TokenResponse, UserOut
from .service import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_COOKIE_KWARGS = dict(httponly=True, samesite="strict", secure=False)


@router.post("/register", response_model=UserOut, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await get_user_by_email(body.email, db)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    return await create_user(body.email, body.password, db)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    user = await get_user_by_email(body.email, db)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(user["id"])
    refresh_token = create_refresh_token(user["id"])
    response.set_cookie(key="access_token", value=access_token, **_COOKIE_KWARGS)
    response.set_cookie(key="refresh_token", value=refresh_token, **_COOKIE_KWARGS)
    return TokenResponse(access_token=access_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No refresh token")
    try:
        payload = decode_token(token)
        if payload.get("type") != "refresh":
            raise ValueError
        user_id = payload["sub"]
    except (JWTError, ValueError, KeyError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    user = await get_user_by_id(user_id, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    access_token = create_access_token(user_id)
    response.set_cookie(key="access_token", value=access_token, **_COOKIE_KWARGS)
    return TokenResponse(access_token=access_token)


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return {"ok": True}
