import logging
from datetime import datetime
from sqlalchemy import select
from fastapi import HTTPException
from .models import ModelBatch,ModelPointer,SoilingSurvey
from .field_service import require,audit
from . import power_model

def get_pointer(db,lock=False):
    query=select(ModelPointer).where(ModelPointer.id==1)
    if lock:query=query.with_for_update().execution_options(populate_existing=True)
    return db.scalar(query)

def active_batch(db):
    pointer=get_pointer(db)
    if not pointer or not pointer.active_batch_id:return None
    return db.get(ModelBatch,pointer.active_batch_id)

def active_evidence(db):
    pointer=get_pointer(db);batch=active_batch(db)
    if not batch:raise HTTPException(409,'没有启用的参考批次，请先登记并启用')
    return power_model.evidence_for(batch,pointer.epoch)

def register(db,body,user,queue):
    if db.scalar(select(ModelBatch).where(ModelBatch.code==body.code)):raise HTTPException(409,'批次编号已存在')
    rows=[r.model_dump() for r in body.samples]
    batch=ModelBatch(code=body.code,note=body.note,samples=rows,sample_rows=len(rows),sha256=power_model.samples_sha(rows),state='training',created_by=user.email)
    db.add(batch);audit(db,user,f'登记参考批次 {body.code}，样本{len(rows)}条，进入训练');db.commit()
    queue.rpush('solar:training',str(batch.id));return batch

def retrain(db,key,user,queue):
    batch=require(db,ModelBatch,key,True)
    if batch.state!='failed':raise HTTPException(409,'仅训练失败的批次可重新训练')
    batch.state='training';batch.error=''
    audit(db,user,f'重新训练参考批次 {batch.code}');db.commit()
    queue.rpush('solar:training',str(batch.id));return batch

def run_training(db,key):
    batch=require(db,ModelBatch,key,True)
    if batch.state!='training':return batch  # 重复训练消息幂等跳过
    try:
        model,metrics,test_x,test_y=power_model.train_candidate(batch.samples)
        try:power_model.save_artifact(batch.id,model)
        except OSError:logging.warning('artifact save failed for batch %s',batch.id)
        pointer=get_pointer(db)
        current=db.get(ModelBatch,pointer.active_batch_id) if pointer and pointer.active_batch_id else None
        if current and current.id!=batch.id:
            try:  # 现用模型在同一冻结留出集上评估，仅作比较，失败不影响候选
                batch.current_holdout_mae_kw=power_model.holdout_mae(power_model.load_model(current),test_x,test_y)
                batch.compared_batch_id=current.id
            except Exception:logging.exception('current-model comparison failed for batch %s',batch.id)
        batch.train_rows=metrics['train_rows'];batch.holdout_rows=metrics['holdout_rows']
        batch.train_mae_kw=metrics['train_mae_kw'];batch.holdout_mae_kw=metrics['holdout_mae_kw']
        batch.feature_importance=metrics['feature_importance']
        batch.state='candidate';batch.error=''
    except Exception as error:
        # 训练失败保留现用版本：只标记本批次，不触碰启用指针
        batch.state='failed';batch.error=str(error)[:380]
    db.commit();return batch

def activate(db,key,user,queue):
    batch=require(db,ModelBatch,key,True)
    pointer=get_pointer(db,lock=True)
    if not pointer:pointer=ModelPointer(id=1,epoch=0);db.add(pointer);db.flush()
    if batch.state=='active' and pointer.active_batch_id==batch.id:return batch  # 重试不重复切换
    busy=db.scalar(select(ModelBatch).where(ModelBatch.state=='activating',ModelBatch.id!=batch.id))
    if busy:raise HTTPException(409,f'批次 {busy.code} 正在启用中，请稍后再试')
    if batch.state=='candidate':
        old=db.get(ModelBatch,pointer.active_batch_id) if pointer.active_batch_id else None
        pointer.active_batch_id=batch.id;pointer.epoch+=1
        batch.state='activating';batch.activated_at=datetime.utcnow()
        if old and old.state=='active':old.state='superseded'
        # 以启用为界：同一事务内将未派工巡测的旧估损置为待重算；已派工（scheduled及以后）保留原依据
        swept=0
        stale=db.scalars(select(SoilingSurvey).where(SoilingSurvey.state=='assessed').with_for_update().execution_options(populate_existing=True)).all()
        for survey in stale:
            if survey.state=='assessed' and (survey.model_evidence or {}).get('epoch',0)<pointer.epoch:
                survey.state='evaluating';swept+=1
        audit(db,user,f'启用参考批次 {batch.code}（纪元{pointer.epoch}），{swept}条未派工估损失效待重算')
        db.commit()
    elif batch.state!='activating':raise HTTPException(409,'仅训练完成的候选批次可启用')
    # 启用提交后重算入队；中断恢复与重试走同一路径，重复入队由worker按状态幂等跳过
    for survey_id in db.scalars(select(SoilingSurvey.id).where(SoilingSurvey.state=='evaluating')).all():queue.rpush('solar:assessment',str(survey_id))
    batch=require(db,ModelBatch,key,True)
    if batch.state=='activating':batch.state='active';db.commit()
    return batch

def resume_activations(db,queue):
    for batch in db.scalars(select(ModelBatch).where(ModelBatch.state=='activating')).all():
        for survey_id in db.scalars(select(SoilingSurvey.id).where(SoilingSurvey.state=='evaluating')).all():queue.rpush('solar:assessment',str(survey_id))
        batch.state='active'
    db.commit()

def enqueue_pending(db,queue):
    for survey_id in db.scalars(select(SoilingSurvey.id).where(SoilingSurvey.state=='evaluating')).all():queue.rpush('solar:assessment',str(survey_id))
    for batch_id in db.scalars(select(ModelBatch.id).where(ModelBatch.state=='training')).all():queue.rpush('solar:training',str(batch_id))
