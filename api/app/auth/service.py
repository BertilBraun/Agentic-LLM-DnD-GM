import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_access_token(user_id: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_expire_minutes)
    payload = {'sub': user_id, 'exp': exp, 'type': 'access'}
    return jwt.encode(payload, settings.jwt_secret, algorithm='HS256')


def create_refresh_token(user_id: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_expire_days)
    payload = {'sub': user_id, 'exp': exp, 'type': 'refresh'}
    return jwt.encode(payload, settings.jwt_secret, algorithm='HS256')


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=['HS256'])


async def create_user(email: str, password: str, db: AsyncSession) -> dict:
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(password)
    await db.execute(
        text('INSERT INTO users (id, email, password_hash) VALUES (:id, :email, :pw_hash)'),
        {'id': user_id, 'email': email, 'pw_hash': pw_hash},
    )
    await db.commit()
    return {'user_id': user_id, 'email': email}


async def get_user_by_id(user_id: str, db: AsyncSession) -> dict | None:
    row = await db.execute(
        text('SELECT id, email FROM users WHERE id = :id'),
        {'id': user_id},
    )
    r = row.mappings().first()
    if r is None:
        return None
    return {'id': str(r['id']), 'email': r['email']}


async def get_user_by_email(email: str, db: AsyncSession) -> dict | None:
    row = await db.execute(
        text('SELECT id, email, password_hash FROM users WHERE email = :email'),
        {'email': email},
    )
    r = row.mappings().first()
    if r is None:
        return None
    return {'id': str(r['id']), 'email': r['email'], 'password_hash': r['password_hash']}
