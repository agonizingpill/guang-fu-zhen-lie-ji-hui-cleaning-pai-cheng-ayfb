import os,time,json,traceback
from pathlib import Path
from sqlalchemy import select
import redis
from .database import Session
from .models import ArrayBlock,SoilingSurvey,ModelBatch,ModelPointer
from . import power_model,batch_service
queue=redis.Redis.from_url(os.getenv('REDIS_URL','redis://redis:6379/0'),decode_responses=True)
ART=Path(os.getenv('ARTIFACTS_DIR','/artifacts'))
def process_assessment(key):
    try:
        with Session() as db:
            # 指针行锁串行化估损写入与批次启用：旧纪元的计算不可能晚于启用提交落库
            pointer=db.scalar(select(ModelPointer).where(ModelPointer.id==1).with_for_update().execution_options(populate_existing=True))
            survey=db.scalar(select(SoilingSurvey).where(SoilingSurvey.id==key).with_for_update().execution_options(populate_existing=True))
            if not survey or survey.state!='evaluating':return
            if not pointer or not pointer.active_batch_id:raise RuntimeError('没有启用的参考批次')
            batch=db.get(ModelBatch,pointer.active_batch_id)
            model=power_model.load_model(batch)
            clean,loss=power_model.predict(model,survey)
            block=db.get(ArrayBlock,survey.block_id)
            survey.clean_kw=clean;survey.predicted_loss=loss;survey.model_evidence=power_model.evidence_for(batch,pointer.epoch)
            survey.recommended_water=round(block.area_m2*.6*(1+.1*survey.dust),1)
            survey.state='assessed';db.commit()
            ART.mkdir(exist_ok=True)
            (ART/f'survey-{key}.json').write_text(json.dumps({'survey_id':key,'clean_kw':clean,'loss_kw':loss,'model':survey.model_evidence}))
    except Exception as error:
        traceback.print_exc()
        with Session() as db:
            survey=db.get(SoilingSurvey,key)
            if survey and survey.state=='evaluating':survey.state='blocked';survey.model_evidence={'error':str(error)};db.commit()
def process_training(key):
    try:
        with Session() as db:batch_service.run_training(db,int(key))
    except Exception:traceback.print_exc()
def main():
    try:
        with Session() as db:batch_service.enqueue_pending(db,queue)
    except Exception:traceback.print_exc()
    while True:
        Path('/tmp/solar-heartbeat').touch()
        job=queue.blpop(['solar:assessment','solar:training'],timeout=3)
        if not job:continue
        name,key=job
        if name=='solar:training':process_training(key)
        else:process_assessment(int(key))
if __name__=='__main__':main()
