import os
import redis
from fastapi import APIRouter,Depends
from sqlalchemy import select
from .database import session
from .auth import current,editor,admin
from .models import ModelBatch
from .schemas import BatchInput
from .field_service import require
from . import batch_service
queue=redis.Redis.from_url(os.getenv('REDIS_URL','redis://redis:6379/0'),decode_responses=True)
router=APIRouter(prefix='/model-batches')
@router.get('')
def batches(user=Depends(current),db=Depends(session)):return db.scalars(select(ModelBatch).order_by(ModelBatch.id.desc())).all()
@router.get('/{key}')
def batch(key:int,user=Depends(current),db=Depends(session)):return require(db,ModelBatch,key)
@router.post('')
def register(body:BatchInput,user=Depends(editor),db=Depends(session)):return batch_service.register(db,body,user,queue)
@router.post('/{key}/train')
def retrain(key:int,user=Depends(editor),db=Depends(session)):return batch_service.retrain(db,key,user,queue)
@router.post('/{key}/activate')
def activate(key:int,user=Depends(admin),db=Depends(session)):return batch_service.activate(db,key,user,queue)
