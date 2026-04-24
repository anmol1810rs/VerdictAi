"""
backend/auth/router.py

GET /auth/me — verifies the Supabase JWT from the Authorization header
               and returns the authenticated user's id and email.

In DEV_MODE the endpoint returns a static dev user so local dev works
without a real Supabase session.
"""
from fastapi import APIRouter, Header, HTTPException
from backend.config import DEV_MODE, SUPABASE_URL, SUPABASE_ANON_KEY

router = APIRouter()


@router.get("/auth/me", tags=["auth"])
def get_me(authorization: str = Header(default="")):
    """
    Verify caller's Supabase JWT and return {user_id, email}.
    Expects:  Authorization: Bearer <access_token>
    """
    if DEV_MODE:
        return {"user_id": "dev-user", "email": "dev@localhost"}

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=503, detail="Auth not configured on server.")

    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        resp = sb.auth.get_user(token)
        user = resp.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        return {"user_id": user.id, "email": user.email}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {exc}") from exc
