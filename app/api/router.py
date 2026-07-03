"""Aggregate API router. Feature routers are mounted here as they are built."""

from fastapi import APIRouter

from app.api import auth_routes, enrollment, events, gallery, photos

api_router = APIRouter()

api_router.include_router(auth_routes.router)
api_router.include_router(events.router)
api_router.include_router(photos.router)
api_router.include_router(enrollment.router)
api_router.include_router(gallery.router)
