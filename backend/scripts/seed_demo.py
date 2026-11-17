import hashlib,json,random
from datetime import datetime,timedelta
from pathlib import Path
from sqlalchemy import select,func
from src.database import Base,engine,Session
from src.models import User,ArrayBlock,SoilingSurvey,WashMission,ModelBatch,ModelPointer
from src.revisit_models import WaterDelivery,FollowupSurvey
from src.auth import hash_password
from src.operations_models import WashCrew
from src import batch_service,power_model

def demo_rows(seed):
    rng=random.Random(seed);rows=[]
    for i in range(200):
        capacity=[80,100,120,150][i%4];irradiance=rng.uniform(350,1050);temperature=rng.uniform(18,65);slope=rng.uniform(10,35)
        clean=capacity*irradiance/1000*(1-.004*(temperature-25))*(.98-.0003*(slope-23)**2)+rng.gauss(0,.65)
        rows.append({'capacity_kw':capacity,'irradiance':round(irradiance,2),'module_temp':round(temperature,2),'slope':round(slope,2),'clean_kw':round(clean,4)})
    return rows

def seed_batch(db,code,note,rows,sha256):
    batch=db.scalar(select(ModelBatch).where(ModelBatch.code==code))
    if batch:return batch
    batch=ModelBatch(code=code,note=note,samples=rows,sample_rows=len(rows),sha256=sha256,state='training',created_by='seed')
    db.add(batch);db.flush();batch_service.run_training(db,batch.id);return batch

def seed():
    Base.metadata.create_all(engine)
    with Session() as db:
        for role in ['admin','planner','viewer']:
            email=role+'@qingling.test'
            if not db.scalar(select(User).where(User.email==email)):db.add(User(email=email,password=hash_password('Solar2026!'),role=role))
        for code,minutes in [('CREW-SOUTH',480),('CREW-RIDGE',360)]:
            if not db.scalar(select(WashCrew).where(WashCrew.code==code)):db.add(WashCrew(code=code,max_daily_minutes=minutes))
        blocks=[('SL-A01','南坡一号','南坡',150,23,780,1500),('SL-A02','南坡二号','南坡',120,25,630,1200),('SL-B01','脊线一号','脊线',100,18,520,900),('SL-C01','西坡一号','西坡',80,30,420,600)]
        for code,name,sector,cap,slope,area,water in blocks:
            if not db.scalar(select(ArrayBlock).where(ArrayBlock.code==code)):db.add(ArrayBlock(code=code,name=name,sector=sector,capacity_kw=cap,slope=slope,area_m2=area,water_liters=water))
        db.flush();rows=db.scalars(select(ArrayBlock).order_by(ArrayBlock.id)).all()
        measurements=[('SV-1117-A01',rows[0],800,35,18,3.2,88),('SV-1117-A02',rows[1],820,33,12,2.1,76),('SV-CLOUD-01',rows[2],240,23,8,1.2,18)]
        for code,block,irr,temp,rain,dust,power in measurements:
            if not db.scalar(select(SoilingSurvey).where(SoilingSurvey.code==code)):db.add(SoilingSurvey(code=code,block_id=block.id,capacity_kw=block.capacity_kw,slope=block.slope,irradiance=irr,module_temp=temp,rain_days=rain,dust=dust,observed_kw=power,state='recorded'))
        db.flush()
        block=db.scalar(select(ArrayBlock).where(ArrayBlock.code=='SL-C01'))
        if not db.scalar(select(WaterDelivery).where(WaterDelivery.delivery_code=='WD-DEMO-01')):
            db.add(WaterDelivery(block_id=block.id,delivery_code='WD-DEMO-01',liters=500,source='西坡蓄水池 / 水车02'))
        historical=db.scalar(select(SoilingSurvey).where(SoilingSurvey.code=='SV-DEMO-CLEAN'))
        if not historical:
            historical=SoilingSurvey(code='SV-DEMO-CLEAN',block_id=block.id,capacity_kw=block.capacity_kw,slope=block.slope,irradiance=800,module_temp=35,rain_days=15,dust=2,observed_kw=50,clean_kw=62,predicted_loss=12,recommended_water=100,state='closed')
            db.add(historical);db.flush()
        mission=db.scalar(select(WashMission).where(WashMission.survey_id==historical.id))
        if not mission:
            mission=WashMission(survey_id=historical.id,water_liters=100,available_liters=700,status='accepted',post_kw=62,post_irradiance=800,normalized_kw=62,verified_gain=12,dispatched_at=datetime.utcnow()-timedelta(days=7),accepted_at=datetime.utcnow()-timedelta(days=7))
            db.add(mission);db.flush()
        if not db.scalar(select(FollowupSurvey).where(FollowupSurvey.mission_id==mission.id,FollowupSurvey.days_after==2)):
            db.add(FollowupSurvey(mission_id=mission.id,days_after=2,irradiance=800,observed_kw=61,normalized_kw=61,normalized_loss=100/62,state='stable',baseline_snapshot={'post_kw':62,'post_irradiance':800,'accepted_at':mission.accepted_at.isoformat(),'block_id':block.id,'block_code':block.code,'survey_id':historical.id,'survey_code':historical.code,'threshold_pct':5}))
        raw=(Path(__file__).resolve().parents[1]/'fixtures/clean_reference.json').read_bytes()
        baseline=seed_batch(db,'RB-BASE-2026','内置200条固定seed42清洁参考样本',json.loads(raw),hashlib.sha256(raw).hexdigest())
        if baseline.state in('training','failed'):baseline.state='training';db.flush();batch_service.run_training(db,baseline.id)
        if baseline.state=='candidate':baseline.state='active';baseline.activated_at=datetime.utcnow()
        if not db.scalar(select(ModelPointer).where(ModelPointer.id==1)):db.add(ModelPointer(id=1,active_batch_id=baseline.id,epoch=1))
        demo=demo_rows(43)
        seed_batch(db,'RB-DEMO-2027','演示用新一批参考样本（seed43同口径模拟）',demo,power_model.samples_sha(demo))
        db.commit();print({key:db.scalar(select(func.count()).select_from(model)) for key,model in [('users',User),('blocks',ArrayBlock),('surveys',SoilingSurvey)]})
if __name__=='__main__':seed()
